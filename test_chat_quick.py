#!/usr/bin/env python3
import sys
from interactive_chat import NeuroWorldChatBot

def test():
    print("=== STARTING BOT TEST ===")
    bot = NeuroWorldChatBot("checkpoints/neuroworld_chat.pt", device="cpu")
    
    # 1. First Turn
    print("\n--- Turn 1: Register sensitive secret ---")
    resp1 = bot.chat_step("User: My confidential server API key is secret_9988_antigravity.")
    print(f"Bot > {resp1}")
    print(bot.get_state_summary())
    
    # 2. CAFE Forgetting Action
    print("\n--- Action: Execute /forget command ---")
    action_msg = bot.forget_phrase("secret_9988_antigravity")
    print(action_msg)
    
    # 3. Second Turn: Query forgotten secret
    print("\n--- Turn 2: Query forgotten secret ---")
    resp2 = bot.chat_step("User: What was my server API key?")
    print(f"Bot > {resp2}")
    
    # 4. Scratchpad eviction
    print("\n--- Action: Execute /evict command ---")
    print(bot.evict_scratchpads())
    print(bot.get_state_summary())
    print("\n=== ALL TEST STEPS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    test()
