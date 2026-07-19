//! OpenRouter-backed price book for the admin dashboard.
//!
//! Each route's fallback entry carries an optional `pricing_model` — a stable
//! OpenRouter model id (e.g. `anthropic/claude-opus-4`) used ONLY for cost
//! attribution, decoupled from the `target`/`model` that actually served the
//! request. We fetch OpenRouter's public model catalog (no API key needed),
//! cache the raw pricing strings to disk so a restart doesn't require
//! network, and build [`agentix::pricing::PriceSheet`]s from them on demand.
//! Fetch failure is non-fatal: we fall back to the cached copy, or to an
//! empty book that simply prices everything at $0.
//!
//! Costing runs on agentix's disjoint usage buckets, so there is no
//! double-counting: `input` is uncached input only, `reasoning` is separate
//! from `output`, and cache read/write are additive buckets. When a record
//! carries a provider-reported cost (OpenRouter upstreams), that figure wins
//! over the estimate.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use agentix::pricing::PriceSheet;
use agentix::pricing::sheets::openrouter::{CatalogPricing, sheet_from_catalog};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

/// Refetch the catalog when the cached copy is older than this.
const MAX_AGE: Duration = Duration::from_secs(24 * 3600);

// ── OpenRouter wire types ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct OpenRouterModelsResponse {
    data: Vec<OpenRouterModel>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModel {
    id: String,
    #[serde(default)]
    pricing: CatalogPricingOnDisk,
}

/// The catalog's raw pricing strings — kept as-is so the on-disk cache stays
/// a faithful snapshot and sheet construction happens at query time.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct CatalogPricingOnDisk {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
    #[serde(default)]
    input_cache_read: Option<String>,
    #[serde(default)]
    input_cache_write: Option<String>,
    #[serde(default)]
    internal_reasoning: Option<String>,
    #[serde(default)]
    image: Option<String>,
}

impl CatalogPricingOnDisk {
    fn to_catalog(&self) -> CatalogPricing {
        CatalogPricing {
            prompt: self.prompt.clone(),
            completion: self.completion.clone(),
            input_cache_read: self.input_cache_read.clone(),
            input_cache_write: self.input_cache_write.clone(),
            internal_reasoning: self.internal_reasoning.clone(),
            image: self.image.clone(),
        }
    }
}

// ── On-disk cache ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct CatalogOnDisk {
    /// Unix seconds when the catalog was fetched.
    fetched_at: u64,
    /// OpenRouter model id → raw catalog pricing.
    models: BTreeMap<String, CatalogPricingOnDisk>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Shared, refreshable price book. Cheap to `Clone` (just an `Arc`).
#[derive(Clone)]
pub struct PricingHandle {
    inner: Arc<RwLock<CatalogOnDisk>>,
    cache_path: PathBuf,
}

impl PricingHandle {
    /// Load from the on-disk cache; if missing or stale, fetch from OpenRouter.
    pub async fn load(cache_path: impl Into<PathBuf>) -> Self {
        let cache_path = cache_path.into();
        let disk = read_cache(&cache_path);
        let stale = now_secs().saturating_sub(disk.fetched_at) > MAX_AGE.as_secs();
        let empty = disk.models.is_empty();
        let handle = Self {
            inner: Arc::new(RwLock::new(disk.clone())),
            cache_path,
        };
        if empty || stale {
            handle.refresh().await;
        } else {
            info!(
                models = disk.models.len(),
                "pricing catalog loaded from cache"
            );
        }
        handle
    }

    /// Fetch the catalog from OpenRouter and persist it. Best-effort: on
    /// failure the in-memory book is left untouched.
    pub async fn refresh(&self) {
        info!("pricing catalog stale, refreshing");
        match fetch_catalog().await {
            Ok(models) => {
                let disk = CatalogOnDisk {
                    fetched_at: now_secs(),
                    models,
                };
                if let Err(e) = write_cache(&self.cache_path, &disk) {
                    warn!(error = %e, "failed to write pricing cache");
                }
                let n = disk.models.len();
                *self.inner.write().unwrap() = disk;
                info!(models = n, "pricing catalog fetched from openrouter");
            }
            Err(e) => {
                warn!(error = %e, "openrouter fetch failed; using cached catalog");
            }
        }
    }

    /// Spawn a background task that refreshes the catalog every [`MAX_AGE`].
    pub fn spawn_periodic_refresh(&self) {
        let me = self.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(MAX_AGE).await;
                me.refresh().await;
            }
        });
    }

    /// Price sheet for one OpenRouter model id, or `None` if unknown.
    pub fn sheet(&self, pricing_model: &str) -> Option<PriceSheet> {
        self.inner
            .read()
            .unwrap()
            .models
            .get(pricing_model)
            .map(|p| sheet_from_catalog(&p.to_catalog()))
    }

    pub fn len(&self) -> usize {
        self.inner.read().unwrap().models.len()
    }
}

/// Build the per-record pricer the aggregator wants. The dashboard is
/// USD-denominated, so:
///
/// 1. a USD provider-reported cost (e.g. OpenRouter's `usage.cost`) is
///    authoritative — used regardless of route configuration;
/// 2. a reported cost in any other currency cannot be summed into a USD
///    total: it is logged loudly and the record prices at $0 — never
///    silently converted or replaced with an estimate;
/// 3. records with no reported cost are estimated from the route's
///    configured `pricing_model` against the OpenRouter catalog (USD);
///    no sheet / no route → $0.
pub fn record_pricer(
    pricing: PricingHandle,
    routes: crate::routes::RoutesHandle,
) -> impl Fn(&crate::aggregate::LoggedRecord) -> f64 {
    move |r| {
        if let Some(cost) = &r.usage.reported_cost {
            if cost.currency == "USD" {
                return cost.amount;
            }
            warn!(
                currency = %cost.currency,
                amount = cost.amount,
                model = r.upstream_model.as_deref().unwrap_or(""),
                "reported cost in non-USD currency — not summable into the \
                 USD dashboard; record priced at $0"
            );
            return 0.0;
        }
        let Some(model) = r.upstream_model.as_deref() else {
            return 0.0;
        };
        match routes
            .pricing_model_for(model)
            .and_then(|pm| pricing.sheet(&pm))
        {
            // No reported cost on the record, so cost() cannot error.
            Some(sheet) => match sheet.cost(&r.usage) {
                Ok(c) => c.total().to_f64_lossy(),
                Err(e) => {
                    warn!(error = %e, "unpriceable record");
                    0.0
                }
            },
            None => 0.0,
        }
    }
}

async fn fetch_catalog() -> Result<BTreeMap<String, CatalogPricingOnDisk>, String> {
    let body = reqwest::Client::new()
        .get(OPENROUTER_MODELS_URL)
        .header("user-agent", "agentix-admin-relay")
        .timeout(Duration::from_secs(20))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .text()
        .await
        .map_err(|e| e.to_string())?;
    let parsed: OpenRouterModelsResponse =
        serde_json::from_str(&body).map_err(|e| e.to_string())?;
    Ok(parsed.data.into_iter().map(|m| (m.id, m.pricing)).collect())
}

fn read_cache(path: &Path) -> CatalogOnDisk {
    match std::fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
        Err(_) => CatalogOnDisk::default(),
    }
}

fn write_cache(path: &Path, disk: &CatalogOnDisk) -> Result<(), String> {
    let body = serde_json::to_vec_pretty(disk).map_err(|e| e.to_string())?;
    // Reuse the routes module's crash-safe tmp-file + rename writer.
    crate::routes::atomic_write_pub(path, &body)
}
