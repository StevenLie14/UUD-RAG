"""
User Interface Helper
Clean interface utilities for user interactions
"""

from typing import List


class UserInterface:
    """Clean user interface helper"""
    
    @staticmethod
    def print_header(title: str):
        """Print section header"""
        print("\n" + "="*80)
        print(title.center(80))
        print("="*80)
    
    @staticmethod
    def print_subheader(title: str):
        """Print subsection header"""
        print("\n" + "-"*80)
        print(title)
        print("-"*80)
    
    @staticmethod
    def get_choice(prompt: str, options: List[str], default: str = "1") -> str:
        """Get user choice from options"""
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        
        choice = input(f"\nEnter choice (1-{len(options)}): ").strip()
        return choice if choice else default
    
    @staticmethod
    def confirm(prompt: str, default: bool = True) -> bool:
        """Get yes/no confirmation"""
        default_str = "Y/n" if default else "y/N"
        choice = input(f"\n{prompt} ({default_str}): ").strip().lower()
        
        if not choice:
            return default
        return choice == 'y'
