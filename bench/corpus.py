"""Benchmark dataset: corpus, queries, and a hand-labeled sanity set.

CORPUS  : list of (id, text, topic) documents to search over.
QUERIES : list of (query_text, expected_topic) search queries.
SANITY  : query_text -> set of doc ids that are *obviously* relevant.
          Used to confirm the full-dim reference ranking is itself sane.
"""

# (id, text, topic)
CORPUS = [
    # --- programming ---
    ("p1", "Use try/except blocks to handle runtime errors gracefully in Python.", "programming"),
    ("p2", "Git lets you track changes and roll back to any previous commit.", "programming"),
    ("p3", "A REST API exposes resources over HTTP using GET, POST, PUT, DELETE.", "programming"),
    ("p4", "Unit tests verify that small pieces of code behave as expected.", "programming"),
    ("p5", "Docker packages an application and its dependencies into a container.", "programming"),
    ("p6", "A linked list stores elements in nodes that point to the next node.", "programming"),

    # --- finance ---
    ("f1", "Compound interest grows your savings faster the longer you invest.", "finance"),
    ("f2", "Diversifying a portfolio reduces risk by spreading investments.", "finance"),
    ("f3", "Inflation erodes the purchasing power of cash held over time.", "finance"),
    ("f4", "A credit score reflects how reliably you repay borrowed money.", "finance"),
    ("f5", "Index funds track a market index and charge low management fees.", "finance"),
    ("f6", "A budget plans how income is split between spending and saving.", "finance"),

    # --- cooking ---
    ("c1", "Searing meat at high heat locks in flavour through the Maillard reaction.", "cooking"),
    ("c2", "Let bread dough rise until it doubles before baking.", "cooking"),
    ("c3", "Add salt to pasta water to season the noodles from the inside.", "cooking"),
    ("c4", "Caramelising onions slowly brings out their natural sweetness.", "cooking"),
    ("c5", "Resting a roast lets the juices redistribute before carving.", "cooking"),
    ("c6", "Whisk eggs and sugar until pale to make a light sponge cake.", "cooking"),

    # --- animals ---
    ("a1", "Dogs are loyal companions and thrive on daily exercise.", "animals"),
    ("a2", "Cats groom themselves and sleep up to sixteen hours a day.", "animals"),
    ("a3", "Honeybees communicate the location of flowers through a waggle dance.", "animals"),
    ("a4", "Elephants have remarkable long-term memory and strong social bonds.", "animals"),
    ("a5", "Penguins huddle together to conserve heat in the Antarctic cold.", "animals"),
    ("a6", "Octopuses can change colour and texture to blend into surroundings.", "animals"),

    # --- sports ---
    ("s1", "A marathon is a long-distance race covering 42.195 kilometres.", "sports"),
    ("s2", "In cricket a batsman scores runs while protecting the wicket.", "sports"),
    ("s3", "Tennis players win a set by reaching six games with a two-game lead.", "sports"),
    ("s4", "Strength training builds muscle through progressive overload.", "sports"),
    ("s5", "A football team scores by getting the ball into the opponent goal.", "sports"),
    ("s6", "Swimmers reduce drag with streamlined body positions and turns.", "sports"),

    # --- health ---
    ("h1", "Drinking enough water supports digestion and concentration.", "health"),
    ("h2", "Regular sleep of seven to nine hours improves memory and mood.", "health"),
    ("h3", "A balanced diet includes proteins, carbohydrates, fats, and fibre.", "health"),
    ("h4", "Consistent medication adherence is key to managing chronic conditions.", "health"),
    ("h5", "Mindful breathing can lower stress and calm the nervous system.", "health"),
]

# (query_text, expected_topic)
QUERIES = [
    ("how do I catch and handle errors in my code", "programming"),
    ("ways to roll back changes in version control", "programming"),
    ("how to grow my long-term savings", "finance"),
    ("reducing investment risk", "finance"),
    ("getting more flavour when cooking meat", "cooking"),
    ("facts about how animals stay warm or hidden", "animals"),
    ("rules of a racket sport", "sports"),
    ("how much sleep should I get", "health"),
    ("staying consistent with my treatment", "health"),
]

# query_text -> set of doc ids that are obviously relevant (sanity check only)
SANITY = {
    "how do I catch and handle errors in my code": {"p1"},
    "ways to roll back changes in version control": {"p2"},
    "how to grow my long-term savings": {"f1", "f5"},
    "how much sleep should I get": {"h2"},
}


if __name__ == "__main__":
    topics = {}
    for _id, _text, topic in CORPUS:
        topics[topic] = topics.get(topic, 0) + 1
    print(f"{len(CORPUS)} docs across {len(topics)} topics: "
          + ", ".join(f"{t}={n}" for t, n in topics.items()))
    print(f"{len(QUERIES)} queries")
    print(f"{len(SANITY)} sanity-labeled queries")
