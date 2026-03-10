Inspired by a16z-infra/ai-town, I am building a tile-based fantasy market simulation where autonomous AI agents live and trade within a persistent physical world. I aim to move beyond simple chat by creating a logic-driven backend where every action—from moving across the map to buying a legendary item—is governed by spatial coordinates and real-time inventory management. My intention is to provide a comprehensive learning experience for users by demonstrating these agentic scenarios and allowing them to create their own player character that is powered by a custom AI agent to achieve specific goals.

I intend to empower individual agents to act as goal-oriented guards and shopkeepers who navigate obstacles, take actions, and negotiate transactions without direct human intervention. This project demonstrates the power of agentic systems by creating a living environment where the winning condition for my player agent is achieved through successful navigation and strategic interaction with NPC-managed shops.

In the project's workflow, the simulation functions as an autonomous, persistent environment that initializes a populated world state by generating tile-based shops, stocking inventories, and spawning shopkeeper NPCs alongside mobile guard entities governed by patrol logic. The system operates as a "headless" live simulation where agentic shopkeepers and guards continuously update their states and behaviors based on the world’s heartbeat, independent of external user input. A player is user defined and provided instruction as a learning experience. When a user-defined player agent joins the session, it is injected into the world state as an autonomous actor that utilizes an LLM-driven reasoning loop to observe the spatial grid, identify the target high-priced item, and navigate the map to execute a goal-oriented strategy. This player agent operates by consuming raw world data it can realisticlly observe—such as guard positions and shopkeeper inventory—and outputting discrete actions like movement, dialogue, attacks, or transactions, allowing it to adapt dynamically to the environment's real-time changes to achieve its objective. 

## ARCHITECTURE
ai-market-sim/
├── data/
│   ├── map.json             # The grid data (walls, floors, shops)
│   ├── map.py               # Map generation utilities
│   └── items.json           # The catalog of fantasy goods (to be created)
├── src/
│   ├── models/              
│   │   └── schema.py        # Blueprints for Tiles, Items, and Actors
│   ├── simulation/          
│   │   ├── physics.py       # Movement rules and collision detection
│   │   └── engine.py        # The "Heartbeat" loop that updates the world
│   ├── agents/              
│   │   ├── brain.py         # Smolagents logic and tool registration
│   │   └── prompts.py       # LLM personality instructions
│   └── ui/                  
│       ├── map_view.py      # NiceGUI map visualization component
│       └── log_feed.py      # Real-time action log display
├── main.py                  # FastAPI + NiceGUI web interface entry point
├── Dockerfile               
└── requirements.txt         


# 🗺️ ai-market-sim: Implementation Roadmap

Follow this order to build your project from the ground up. Each step provides the foundation for the next.

---

### 🏗️ Phase 1: The Foundation (Data & Rules)
*Goal: Define what the world is and what the rules are.*

1.  **`src/models/schema.py`** * **Task:** Define Pydantic classes for `Item`, `Tile`, `Actor`, and `Map`.  
    * **Why:** This is your "Source of Truth" that ensures all data follows a strict format.

2.  **`data/map.json`** * **Task:** Create a manual JSON file representing a grid with walls, floors, and shops.  
    * **Why:** Your code needs a physical environment to "load" into memory to test logic.

3.  **`src/simulation/physics.py`** * **Task:** Write functions like `is_walkable(x, y)` and implement features like actor's field of vision.
    * **Why:** These are the "Laws of the Universe" that prevent actors from walking through walls and seeing through them.

---

### ⚙️ Phase 2: The Living World (Mechanics)
*Goal: Make the world move and update without AI yet.*

4.  **`src/simulation/engine.py`** * **Task:** Create the "Heartbeat" loop. Implement simple non-AI movement (e.g., a Guard moving back and forth).  
    * **Why:** To ensure your state updates correctly every second/tick.

5.  **`src/ui/map_view.py` & `src/ui/log_feed.py`** * **Task:** Integrate NiceGUI components to visualize the grid using ASCII characters and display live action logs.  
    * **Why:** You need a way to visually verify that Phase 1 and 2 are working correctly in real-time.

---

### 🧠 Phase 3: The Agentic Brain (The AI)
*Goal: Give the characters the ability to reason and act.*

6.  **`src/agents/prompts.py`** * **Task:** Define the System Instructions for different roles (The Greedy Merchant, The Stoic Guard).  
    * **Why:** To give the LLM a personality and a specific goal for each entity.

7.  **`src/agents/brain.py`** * **Task:** Integrate **Smolagents**. Map your Physics/Action functions to "AI Tools."  
    * **Why:** This connects the LLM to your world so it can say "I want to move North" and the code executes it.

---

### 🌐 Phase 4: Opening the Gates (Connectivity)
*Goal: Allow the outside world to see and join the simulation.*

8.  **`main.py`** * **Task:** Integrate **NiceGUI** with FastAPI to create a web interface. Set up endpoints (e.g., `GET /map`, `POST /player/action`).  
    * **Why:** This provides real-time visual feedback and allows users to interact with the simulation through a browser.

9.  **`Dockerfile` & `requirements.txt`** * **Task:** Package the application into a container.  
    * **Why:** To fulfill your goal of an "educational experience" that anyone can run with one command.