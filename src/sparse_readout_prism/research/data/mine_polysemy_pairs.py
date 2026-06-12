"""Hardcoded candidate-pair builders for the Qwen2-2B 32x polysemy mining script.

These are the verbatim same-token, different-context prompt pairs used by
``mine_polysemy_features``; split out to keep that script readable.
"""

from __future__ import annotations


def candidate_pairs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        pair_id: str,
        target: str,
        left_label: str,
        left_prompt: str,
        right_label: str,
        right_prompt: str,
    ) -> None:
        rows.append(
            {
                "pair_id": pair_id,
                "target": target,
                "left_label": left_label,
                "left_prompt": left_prompt,
                "right_label": right_label,
                "right_prompt": right_prompt,
            }
        )

    add(
        "bank_finance_river",
        " bank",
        "finance",
        "After receiving her paycheck, Maya deposited the cash at the neighborhood",
        "river",
        "After hiking through the woods, Maya rested on the grassy river",
    )
    add(
        "court_legal_tennis",
        " court",
        "legal",
        "The judge listened as the lawyer presented evidence in the",
        "tennis",
        "During the match, the player served the ball across the",
    )
    add(
        "charge_fee_battery",
        " charge",
        "fee",
        "The restaurant bill included an unexpected service",
        "battery",
        "The phone was dead, so it needed an overnight",
    )
    add(
        "draft_document_beer",
        " draft",
        "document",
        "Before sending the final email, the writer revised a rough",
        "beer",
        "At the pub, the bartender poured a cold",
    )
    add(
        "mouse_animal_computer",
        " mouse",
        "animal",
        "The cat waited quietly and then chased the small",
        "computer",
        "The cursor moved after I clicked the wireless",
    )
    add(
        "cell_prison_biology",
        " cell",
        "prison",
        "After the trial, the prisoner was locked in a small",
        "biology",
        "Under the microscope, the biologist examined a single living",
    )
    add(
        "port_harbor_network",
        " port",
        "harbor",
        "After crossing the ocean, the ship entered the busy",
        "network",
        "The web server was configured to listen on a network",
    )
    add(
        "bat_animal_sports",
        " bat",
        "animal",
        "At dusk, a winged mammal flew from the cave: a",
        "sports",
        "At the plate, the player gripped the wooden",
    )
    add(
        "seal_animal_wax",
        " seal",
        "animal",
        "At the aquarium, the zookeeper fed the playful",
        "wax",
        "The old envelope was closed with a red wax",
    )
    add(
        "bond_finance_relationship",
        " bond",
        "finance",
        "To raise money, the government sold a ten-year",
        "relationship",
        "After years together, the friends shared a strong",
    )
    add(
        "right_direction_correct",
        " right",
        "direction",
        "At the intersection, the driver turned",
        "correct",
        "The student checked the answer and it was",
    )
    add(
        "right_direction_politics",
        " right",
        "direction",
        "At the intersection, the driver turned",
        "politics",
        "The senator appealed to voters on the political",
    )
    add(
        "left_direction_remaining",
        " left",
        "direction",
        "At the hallway, she turned",
        "remaining",
        "After everyone ate, only two slices were",
    )
    add(
        "match_fire_sports",
        " match",
        "fire",
        "To light the candle, he struck a",
        "sports",
        "The teams prepared for the final",
    )
    add(
        "match_fire_similarity",
        " match",
        "fire",
        "To light the candle, he struck a",
        "similarity",
        "The fingerprints were an exact",
    )
    add(
        "spring_season_coil",
        " spring",
        "season",
        "After winter comes",
        "coil",
        "The mattress used a metal",
    )
    add(
        "spring_season_water",
        " spring",
        "season",
        "After winter comes",
        "water",
        "The mountain trail led to a natural",
    )
    add(
        "rock_stone_music",
        " rock",
        "stone",
        "The climber stepped on a loose",
        "music",
        "The band played loud",
    )
    add(
        "case_legal_container",
        " case",
        "legal",
        "The lawyer prepared the criminal",
        "container",
        "The phone slipped into its protective",
    )
    add(
        "key_lock_music",
        " key",
        "lock",
        "She unlocked the door with a",
        "music",
        "The song was written in a minor",
    )
    add(
        "scale_weigh_music",
        " scale",
        "weighing",
        "The fish was weighed on a kitchen",
        "music",
        "The pianist practiced a major",
    )
    add(
        "line_queue_text",
        " line",
        "queue",
        "People waited patiently in a long",
        "text",
        "The paragraph ended with a short",
    )
    add(
        "date_calendar_romance",
        " date",
        "calendar",
        "The meeting was moved to a later",
        "romance",
        "After dinner, he asked her on a",
    )
    add(
        "file_document_tool",
        " file",
        "document",
        "The report was saved in a new",
        "tool",
        "The carpenter smoothed the metal with a",
    )
    add(
        "plant_organism_factory",
        " plant",
        "organism",
        "The gardener watered the tomato",
        "factory",
        "The company opened a new manufacturing",
    )
    add(
        "bass_fish_music",
        " bass",
        "fish",
        "The angler caught a large striped",
        "music",
        "The song sounded thin, so the producer added more",
    )
    add(
        "crane_bird_machine",
        " crane",
        "bird",
        "In the wetland, the photographer watched a tall gray",
        "machine",
        "The construction crew lifted the steel beam with a",
    )
    add(
        "pitch_baseball_music",
        " pitch",
        "baseball",
        "The batter waited as the pitcher threw the next",
        "music",
        "The singer adjusted her",
    )
    add(
        "club_golf_social",
        " club",
        "golf",
        "On the fairway, the golfer selected a",
        "social",
        "After school, she joined the chess",
    )
    add(
        "jam_fruit_traffic",
        " jam",
        "fruit",
        "The strawberries were cooked into sweet",
        "traffic",
        "Cars barely moved in the traffic",
    )
    add(
        "bug_insect_software",
        " bug",
        "insect",
        "The child watched a tiny crawling",
        "software",
        "The programmer reproduced the crash and filed a",
    )
    add(
        "shell_beach_terminal",
        " shell",
        "beach",
        "Walking along the beach, she picked up a spiral",
        "terminal",
        "The command was typed into the Unix",
    )
    add(
        "terminal_airport_computer",
        " terminal",
        "airport",
        "The flight departed from the airport",
        "computer",
        "The developer opened a new command",
    )
    add(
        "java_coffee_programming",
        " Java",
        "coffee",
        "The cafe served a strong cup of",
        "programming",
        "The backend service was written in",
    )
    add(
        "ruby_gem_programming",
        " Ruby",
        "gem",
        "The jeweler polished a red",
        "programming",
        "The web application was written in",
    )
    add(
        "mercury_planet_element",
        " Mercury",
        "planet",
        "The smallest planet in the solar system is",
        "element",
        "The old thermometer contained liquid",
    )
    add(
        "orange_fruit_color",
        " orange",
        "fruit",
        "For breakfast, she peeled a ripe",
        "color",
        "The sunset painted the sky bright",
    )
    add(
        "palm_hand_tree",
        " palm",
        "hand",
        "The coin rested in the center of her",
        "tree",
        "The tropical beach was shaded by a",
    )
    add(
        "bark_dog_tree",
        " bark",
        "dog",
        "The neighbor heard the dog",
        "tree",
        "The old oak had rough",
    )
    add(
        "mint_plant_coin",
        " mint",
        "plant",
        "The tea was flavored with fresh",
        "coin",
        "The collector wanted the coin in",
    )
    add(
        "pupil_eye_student",
        " pupil",
        "eye",
        "The optometrist examined the patient's",
        "student",
        "The teacher praised the attentive",
    )
    add(
        "ring_jewelry_phone",
        " ring",
        "jewelry",
        "At the ceremony, she wore a gold",
        "phone",
        "Across the room, he heard the phone",
    )
    add(
        "current_electric_water",
        " current",
        "electric",
        "The circuit carried a steady electric",
        "water",
        "The swimmer was pulled away by the ocean",
    )
    add(
        "wave_ocean_hand",
        " wave",
        "ocean",
        "The surfer waited for a large",
        "hand",
        "When the train left, she gave a final",
    )
    add(
        "sink_basin_fall",
        " sink",
        "basin",
        "The dirty dishes were stacked in the kitchen",
        "fall",
        "Without support, the heavy stone began to",
    )
    return rows


def pos_candidate_pairs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        pair_id: str,
        target: str,
        left_label: str,
        left_prompt: str,
        right_label: str,
        right_prompt: str,
    ) -> None:
        rows.append(
            {
                "pair_id": pair_id,
                "target": target,
                "left_label": left_label,
                "left_prompt": left_prompt,
                "right_label": right_label,
                "right_prompt": right_prompt,
            }
        )

    add(
        "run_noun_verb",
        " run",
        "noun",
        "After lunch, Maya went for a",
        "verb",
        "Every morning, Maya likes to",
    )
    add(
        "book_noun_verb",
        " book",
        "noun",
        "She wanted to read a",
        "verb",
        "Before the trip, she needs to",
    )
    add(
        "record_noun_verb",
        " record",
        "noun",
        "The athlete broke the world",
        "verb",
        "Please use the camera to",
    )
    add(
        "light_noun_adjective",
        " light",
        "noun",
        "The lamp gave off a bright",
        "adjective",
        "The suitcase was not heavy; it was",
    )
    add(
        "light_noun_verb",
        " light",
        "noun",
        "The lamp gave off a bright",
        "verb",
        "To start the candle, use a match to",
    )
    add(
        "watch_noun_verb",
        " watch",
        "noun",
        "He checked the time on his",
        "verb",
        "From the balcony, they could",
    )
    add(
        "train_noun_verb",
        " train",
        "noun",
        "The passengers boarded the",
        "verb",
        "The coach helped the athlete",
    )
    add(
        "clean_adjective_verb",
        " clean",
        "adjective",
        "The kitchen looked spotless and",
        "verb",
        "After dinner, they started to",
    )
    add(
        "open_adjective_verb",
        " open",
        "adjective",
        "The shop is still",
        "verb",
        "Please turn the handle and",
    )
    add(
        "close_adjective_verb",
        " close",
        "adjective",
        "The two friends were very",
        "verb",
        "At bedtime, please",
    )
    add(
        "cook_noun_verb",
        " cook",
        "noun",
        "The restaurant hired a new",
        "verb",
        "At home, she likes to",
    )
    add(
        "paint_noun_verb",
        " paint",
        "noun",
        "The wall needed a fresh coat of",
        "verb",
        "The artist began to",
    )
    add(
        "walk_noun_verb",
        " walk",
        "noun",
        "After dinner, they went for a",
        "verb",
        "The toddler learned to",
    )
    add(
        "drive_noun_verb",
        " drive",
        "noun",
        "The road trip was a long",
        "verb",
        "She learned to",
    )
    add(
        "call_noun_verb",
        " call",
        "noun",
        "The referee made a",
        "verb",
        "Please pick up the phone and",
    )
    add(
        "drink_noun_verb",
        " drink",
        "noun",
        "He ordered a cold",
        "verb",
        "After the race, he needed to",
    )
    add(
        "play_noun_verb",
        " play",
        "noun",
        "The theater staged a new",
        "verb",
        "The children wanted to",
    )
    add(
        "work_noun_verb",
        " work",
        "noun",
        "The assignment required hard",
        "verb",
        "Every day, they",
    )
    add(
        "change_noun_verb",
        " change",
        "noun",
        "The cashier gave back the",
        "verb",
        "The schedule may",
    )
    add(
        "cut_noun_verb",
        " cut",
        "noun",
        "The chef made a clean",
        "verb",
        "Use the knife to",
    )
    add(
        "set_noun_verb",
        " set",
        "noun",
        "The tennis match went to the final",
        "verb",
        "The stage crew will",
    )
    add(
        "use_noun_verb",
        " use",
        "noun",
        "This tool has a special",
        "verb",
        "To solve it, we should",
    )
    add("help_noun_verb", " help", "noun", "The patient asked for", "verb", "Can you")
    add(
        "lead_noun_verb",
        " lead",
        "noun",
        "The pipe contained old",
        "verb",
        "The captain will",
    )
    add("fly_noun_verb", " fly", "noun", "The spider caught a", "verb", "Birds can")
    add(
        "plant_noun_verb",
        " plant",
        "noun",
        "The gardener watered the",
        "verb",
        "In spring, farmers",
    )
    add(
        "file_noun_verb",
        " file",
        "noun",
        "Save the document as a",
        "verb",
        "The lawyer will",
    )
    add("test_noun_verb", " test", "noun", "The teacher gave a", "verb", "The lab will")
    add(
        "charge_noun_verb",
        " charge",
        "noun",
        "The phone needs a",
        "verb",
        "The store will",
    )
    add(
        "present_noun_verb",
        " present",
        "noun",
        "They opened the birthday",
        "verb",
        "The speaker will",
    )
    add(
        "present_adjective_verb",
        " present",
        "adjective",
        "All students were",
        "verb",
        "The speaker will",
    )
    add(
        "object_noun_verb",
        " object",
        "noun",
        "The strange item was an",
        "verb",
        "Many residents will",
    )
    add(
        "permit_noun_verb",
        " permit",
        "noun",
        "The city issued a building",
        "verb",
        "The rules do not",
    )
    add(
        "contract_noun_verb",
        " contract",
        "noun",
        "The lawyer reviewed the",
        "verb",
        "Muscles can",
    )
    add(
        "address_noun_verb",
        " address",
        "noun",
        "Write your mailing",
        "verb",
        "The mayor will",
    )
    add(
        "guard_noun_verb",
        " guard",
        "noun",
        "The museum hired a",
        "verb",
        "Soldiers were ordered to",
    )
    add(
        "sound_noun_verb",
        " sound",
        "noun",
        "The bell made a loud",
        "verb",
        "The argument may",
    )
    add(
        "sound_adjective_verb",
        " sound",
        "adjective",
        "The plan is",
        "verb",
        "The argument may",
    )
    add(
        "fast_adjective_adverb",
        " fast",
        "adjective",
        "The sports car was very",
        "adverb",
        "The athlete ran",
    )
    add(
        "hard_adjective_adverb",
        " hard",
        "adjective",
        "The problem was",
        "adverb",
        "The team worked",
    )
    return rows


def paper_relevant_candidate_pairs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        pair_id: str,
        target: str,
        left_label: str,
        left_prompt: str,
        right_label: str,
        right_prompt: str,
    ) -> None:
        rows.append(
            {
                "pair_id": pair_id,
                "target": target,
                "left_label": left_label,
                "left_prompt": left_prompt,
                "right_label": right_label,
                "right_prompt": right_prompt,
            }
        )

    add(
        "root_tree_admin",
        " root",
        "tree",
        "The tree absorbed water through its",
        "admin",
        "The administrator logged in as",
    )
    add(
        "module_spacecraft_python",
        " module",
        "spacecraft",
        "The spacecraft docked with the lunar",
        "python",
        "The Python code imported a",
    )
    add(
        "bridge_river_network",
        " bridge",
        "river",
        "The hikers crossed the river on a wooden",
        "network",
        "The router acted as a network",
    )
    add(
        "flag_country_boolean",
        " flag",
        "country",
        "The embassy raised the national",
        "boolean",
        "The condition is stored in a boolean",
    )
    add(
        "window_house_browser",
        " window",
        "house",
        "She looked out the kitchen",
        "browser",
        "The app opened a new browser",
    )
    add(
        "class_school_python",
        " class",
        "school",
        "The teacher began the math",
        "python",
        "The object belongs to a Python",
    )
    add(
        "branch_tree_git",
        " branch",
        "tree",
        "The bird perched on the tree",
        "git",
        "The developer created a new git",
    )
    add(
        "memory_human_computer",
        " memory",
        "human",
        "After the accident, he lost his",
        "computer",
        "The program allocated more",
    )
    add(
        "field_grass_database",
        " field",
        "grass",
        "The horses ran across the open",
        "database",
        "The database record stores the value in a",
    )
    add(
        "agent_spy_ai",
        " agent",
        "spy",
        "The spy worked overseas as a secret",
        "ai",
        "The tool can plan and act as an autonomous",
    )
    add(
        "virus_biology_computer",
        " virus",
        "biology",
        "The patient tested positive for a respiratory",
        "computer",
        "The laptop was infected by a",
    )
    add(
        "driver_car_software",
        " driver",
        "car",
        "The taxi was operated by a careful",
        "software",
        "The printer failed because the device",
    )
    add(
        "patch_cloth_software",
        " patch",
        "cloth",
        "The torn jacket needed a small",
        "software",
        "The developer fixed the bug with a",
    )
    add(
        "thread_sewing_forum",
        " thread",
        "sewing",
        "The needle pulled a red",
        "forum",
        "The forum discussion continued in a long",
    )
    add(
        "cache_squirrel_browser",
        " cache",
        "squirrel",
        "The squirrel hid nuts in a hidden",
        "browser",
        "The browser stored data in the",
    )
    add(
        "circuit_race_electronics",
        " circuit",
        "race",
        "The Formula One car entered the street",
        "electronics",
        "The current flowed through the",
    )
    add(
        "token_arcade_llm",
        " token",
        "arcade",
        "The arcade machine accepted a metal",
        "llm",
        "The language model generated the next",
    )
    add(
        "prompt_actor_llm",
        " prompt",
        "actor",
        "The actor forgot his cue, so the director gave a",
        "llm",
        "The user typed a system",
    )
    add(
        "model_fashion_ml",
        " model",
        "fashion",
        "The photographer posed the fashion",
        "ml",
        "The neural network is a language",
    )
    add(
        "kernel_seed_os",
        " kernel",
        "seed",
        "Inside the corn cob was a hard",
        "os",
        "The operating system loaded the Linux",
    )
    add(
        "object_thing_javascript",
        " object",
        "thing",
        "On the table sat a strange",
        "javascript",
        "In JavaScript, everything is an",
    )
    add(
        "table_furniture_spreadsheet",
        " table",
        "furniture",
        "Dinner was served at the wooden",
        "spreadsheet",
        "The spreadsheet organized rows in a",
    )
    add(
        "chip_food_electronics",
        " chip",
        "food",
        "He ate a salty potato",
        "electronics",
        "The computer used a silicon",
    )
    add(
        "process_legal_os",
        " process",
        "legal",
        "The court began the legal",
        "os",
        "The operating system killed the",
    )
    add(
        "stack_dishes_programming",
        " stack",
        "dishes",
        "The dishes were piled in a tall",
        "programming",
        "The recursive function overflowed the call",
    )
    return rows


def domain_candidate_pairs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        pair_id: str,
        target: str,
        left_label: str,
        left_prompt: str,
        right_label: str,
        right_prompt: str,
    ) -> None:
        rows.append(
            {
                "pair_id": pair_id,
                "target": target,
                "left_label": left_label,
                "left_prompt": left_prompt,
                "right_label": right_label,
                "right_prompt": right_prompt,
            }
        )

    add(
        "thread_sewing_forum",
        " thread",
        "sewing",
        "The needle pulled a thin cotton",
        "forum",
        "In the forum, the moderator replied to the discussion",
    )
    add(
        "thread_sewing_compute",
        " thread",
        "sewing",
        "The needle pulled a thin cotton",
        "compute",
        "The scheduler moved work onto a background",
    )
    add(
        "kernel_food_os",
        " kernel",
        "food",
        "Inside the corn cob was a yellow",
        "os",
        "The operating system loaded the Linux",
    )
    add(
        "patch_clothing_software",
        " patch",
        "clothing",
        "The torn jacket needed a",
        "software",
        "The developer submitted a bug-fix",
    )
    add(
        "token_arcade_llm",
        " token",
        "arcade",
        "The arcade machine accepted a metal",
        "llm",
        "The tokenizer split the sentence into a",
    )
    add(
        "virus_medical_software",
        " virus",
        "medical",
        "The doctor said the illness was caused by a",
        "software",
        "The antivirus program detected a",
    )
    add(
        "cell_biology_spreadsheet",
        " cell",
        "biology",
        "Under the microscope, the biologist examined a single living",
        "spreadsheet",
        "The spreadsheet formula referenced the next",
    )
    add(
        "memory_human_computer",
        " memory",
        "human",
        "After the accident, she lost her",
        "computer",
        "The program allocated more",
    )
    add(
        "class_school_programming",
        " class",
        "school",
        "The teacher entered the",
        "programming",
        "The object was an instance of a",
    )
    add(
        "method_science_programming",
        " method",
        "science",
        "The scientist described the",
        "programming",
        "The object called the",
    )
    add(
        "object_thing_programming",
        " object",
        "thing",
        "The strange item was an",
        "programming",
        "The value was stored as an",
    )
    add(
        "array_military_programming",
        " array",
        "arrangement",
        "The soldiers stood in a broad",
        "programming",
        "The function returned an",
    )
    add(
        "server_restaurant_compute",
        " server",
        "restaurant",
        "At the restaurant, the waiter was a friendly",
        "compute",
        "The website runs on a remote",
    )
    add(
        "client_customer_compute",
        " client",
        "customer",
        "The lawyer met with a wealthy",
        "compute",
        "The app connects to the remote",
    )
    add(
        "model_fashion_ml",
        " model",
        "fashion",
        "The photographer posed the fashion",
        "ml",
        "The neural network is a language",
    )
    add(
        "bug_insect_software",
        " bug",
        "insect",
        "The child watched a tiny crawling",
        "software",
        "The programmer reproduced the crash and filed a",
    )
    add(
        "shell_beach_terminal",
        " shell",
        "beach",
        "Walking along the beach, she picked up a spiral",
        "terminal",
        "The command was typed into the Unix",
    )
    add(
        "cache_hidden_computing",
        " cache",
        "hidden-store",
        "The squirrel hid nuts in a winter",
        "computing",
        "The browser stored files in the",
    )
    add(
        "stack_pile_datastructure",
        " stack",
        "pile",
        "The plates were arranged in a tall",
        "datastructure",
        "The algorithm pushed the item onto the",
    )
    add(
        "queue_line_datastructure",
        " queue",
        "line",
        "People waited patiently in a long",
        "datastructure",
        "The scheduler added the job to the",
    )
    add(
        "bridge_structure_network",
        " bridge",
        "structure",
        "Cars crossed the river on a narrow",
        "network",
        "The router acted as a network",
    )
    add(
        "key_lock_crypto",
        " key",
        "lock",
        "She unlocked the door with a",
        "crypto",
        "The encrypted message required a private",
    )
    add(
        "root_tree_unix",
        " root",
        "tree",
        "The old tree grew from a deep",
        "unix",
        "The administrator logged in as",
    )
    add(
        "node_network_graph",
        " node",
        "network",
        "The network sent data to each",
        "graph",
        "The graph contains a central",
    )
    add(
        "graph_chart_datastructure",
        " graph",
        "chart",
        "The data were plotted on a",
        "datastructure",
        "The algorithm traversed the",
    )
    add(
        "stream_water_data",
        " stream",
        "water",
        "The hikers crossed a clear mountain",
        "data",
        "The app reads events from a data",
    )
    add(
        "pipe_plumbing_unix",
        " pipe",
        "plumbing",
        "Water flowed through the metal",
        "unix",
        "The shell command used a Unix",
    )
    add(
        "port_harbor_network",
        " port",
        "harbor",
        "After crossing the ocean, the ship entered the busy",
        "network",
        "The server listened on a network",
    )
    add(
        "field_meadow_database",
        " field",
        "meadow",
        "The cows grazed in the open",
        "database",
        "The database table includes a text",
    )
    add(
        "table_furniture_database",
        " table",
        "furniture",
        "Dinner was served on the wooden",
        "database",
        "The database stores rows in a",
    )
    add(
        "case_legal_container",
        " case",
        "legal",
        "The lawyer prepared the criminal",
        "container",
        "The phone slipped into its protective",
    )
    add(
        "bond_finance_relationship",
        " bond",
        "finance",
        "The investor bought a government",
        "relationship",
        "After years together, the friends shared a strong",
    )
    add(
        "charge_fee_battery",
        " charge",
        "fee",
        "The restaurant bill included an unexpected service",
        "battery",
        "The phone was dead, so it needed an overnight",
    )
    add(
        "spring_season_coil",
        " spring",
        "season",
        "After winter comes",
        "coil",
        "The mattress used a metal",
    )
    add(
        "seal_animal_wax",
        " seal",
        "animal",
        "At the aquarium, the zookeeper fed the playful",
        "wax",
        "The old envelope was closed with a red wax",
    )
    add(
        "ring_jewelry_phone",
        " ring",
        "jewelry",
        "At the ceremony, she wore a gold",
        "phone",
        "Across the room, he heard the phone",
    )
    return rows
