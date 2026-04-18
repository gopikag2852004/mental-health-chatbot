import json
import random
import os

# Get project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_json(relative_path):

    path = os.path.join(BASE_DIR, relative_path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Load datasets safely
exercises = load_json("data/support/exercises.json")
music = load_json("data/support/music.json")
activities = load_json("data/support/activities.json")
goals = load_json("data/support/goals.json")
reflections = load_json("data/support/reflections.json")


def recommend_support(mood):

    if mood == "stress":
        e = random.choice(exercises["stress_relief"])
        return f"\n\n🧘 You could try this stress relief exercise: **{e['name']}** – {e['description']}"

    elif mood == "anxiety":
        e = random.choice(exercises["anxiety_relief"])
        return f"\n\n🫁 A breathing technique that may help: **{e['name']}** – {e['description']}"

    elif mood == "sad":
        m = random.choice(music["uplifting"])
        return f"\n\n🎵 Music that may help improve mood: **{m['title']} by {m['artist']}**."

    elif mood == "low":
        m = random.choice(music["calming"])
        return f"\n\n🎧 A calming track you could try: **{m['title']} by {m['artist']}**."

    else:
        choice = random.choice(["activity", "goal", "reflection"])

        if choice == "activity":
            a = random.choice(activities["relaxing"])
            return f"\n\n🌿 A small activity you might try: **{a['activity']}**."

        elif choice == "goal":
            g = random.choice(goals["daily_goals"])
            return f"\n\n🎯 A simple goal for today: **{g['goal']}**."

        else:
            r = random.choice(reflections["self_awareness"])
            return f"\n\n💭 Reflection prompt: **{r['prompt']}**."
