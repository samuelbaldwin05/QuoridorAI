# scripts/play_bot.py
from quoridor.game import QuoridorState
from quoridor.display import render
from agents.bot import HeuristicBot
from scripts.play import parse_input

game = QuoridorState()
bot = HeuristicBot()

print("=== You (P0) vs Bot (P1) ===")
print("Move: 'e5'  |  Fence: 'h a3' or 'v c2'  |  'quit' to exit\n")

while not game.done:
    print(render(game))
    print()

    if game.turn == 0:
        # human turn
        while True:
            try:
                raw = input("P0> ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                exit()

            parsed = parse_input(raw)
            if parsed is None:
                print("Invalid input.")
                continue
            if parsed[0] == "quit":
                print("Goodbye!")
                exit()

            try:
                if parsed[0] == "move":
                    game.move_to(parsed[1], parsed[2])
                else:
                    game.place_fence(parsed[1], parsed[2], parsed[3])
                break
            except ValueError as e:
                print(f"Illegal: {e}")
    else:
        # bot turn
        action = bot.choose_action(game)
        if action[0] == "move":
            print(f"Bot moves to ({action[1]}, {action[2]})")
            game.move_to(action[1], action[2])
        else:
            print(f"Bot places {action[3]} fence at ({action[1]}, {action[2]})")
            game.place_fence(action[1], action[2], action[3])

    print()

print(render(game))
print(f"Player {game.winner} wins!")