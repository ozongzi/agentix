#!/usr/bin/env python3
import time
import sys
import os

class CountdownTimer:
    def __init__(self, seconds=60):
        self.seconds = seconds
        self.start_time = None
        
    def clear_screen(self):
        """清屏"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def format_time(self, seconds):
        """格式化时间显示"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def draw_progress_bar(self, current, total, width=50):
        """绘制进度条"""
        progress = current / total
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        percentage = progress * 100
        return f"[{bar}] {percentage:.1f}%"
    
    def display_countdown(self, remaining):
        """显示倒计时界面"""
        self.clear_screen()
        
        # 标题
        print("\n" + "="*60)
        print(" " * 20 + "🎯 倒计时程序 🎯")
        print("="*60 + "\n")
        
        # 时间显示
        time_str = self.format_time(remaining)
        print(" " * 20 + "⏰ 剩余时间:")
        print(" " * 20 + f"  {time_str}")
        print()
        
        # 进度条
        progress_bar = self.draw_progress_bar(self.seconds - remaining, self.seconds)
        print(" " * 10 + progress_bar)
        print()
        
        # 进度信息
        elapsed = self.seconds - remaining
        print(f"已过去: {self.format_time(elapsed)}")
        print(f"总时长: {self.format_time(self.seconds)}")
        print()
        
        # 艺术效果
        if remaining > 30:
            print(" " * 15 + "🚀 倒计时进行中...")
        elif remaining > 10:
            print(" " * 15 + "⚡ 即将结束！")
        else:
            print(" " * 15 + "🔥 最后冲刺！")
        
        print("\n" + "="*60)
    
    def run(self):
        """运行倒计时"""
        print("倒计时开始！")
        time.sleep(1)
        
        self.start_time = time.time()
        
        for i in range(self.seconds, -1, -1):
            self.display_countdown(i)
            
            if i > 0:
                time.sleep(1)
            else:
                print("\n" + "="*60)
                print(" " * 20 + "🎉 倒计时结束！ 🎉")
                print("="*60)
                print("\n" + " " * 15 + "时间到！任务完成！")
                print(" " * 15 + "🎊 恭喜！ 🎊")
                print()
                
                # 显示一些庆祝效果
                for _ in range(3):
                    print(" " * 10 + "✨" * 10)
                    time.sleep(0.3)
                    self.clear_screen()
                    print(" " * 10 + "🎆" * 10)
                    time.sleep(0.3)
                    self.clear_screen()

def main():
    """主函数"""
    try:
        # 创建一个30秒的倒计时
        timer = CountdownTimer(30)
        timer.run()
        
        # 显示统计信息
        print("\n" + "="*60)
        print("📊 统计信息:")
        print(f"• 总倒计时时长: 30秒")
        print(f"• 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(f"• 结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  倒计时被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":
    main()
