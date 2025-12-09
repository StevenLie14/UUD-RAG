from datetime import datetime

class Logger:
    enabled: bool = True
    
    @staticmethod
    def disable() -> None:
        Logger.enabled = False
        
    @staticmethod
    def enable() -> None:
        Logger.enabled = True
    
    @staticmethod
    def log(message: str) -> None:
        if Logger.enabled:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")