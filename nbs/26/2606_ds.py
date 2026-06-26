"""learning howto: dspy

dspy makes the bet that i am bullish on as well

which is that assuming reasonable rule following by modern day lms
we should not rely on their internal structure to call tools etc.
"""
# %%
msg = "Hello dear sir, do you want to come to the bar for ICML26 tomorrow?"
# %%
import dspy
lm = dspy.LM("openai/gpt-5.4-nano")
dspy.configure(lm=lm)

class ExtractEvent(dspy.Signature):
    """Extract event details from an email."""
    email: str = dspy.InputField()
    event_name: str = dspy.OutputField()
    date: str = dspy.OutputField()

extract = dspy.Predict(ExtractEvent)
print(extract(email=msg))
# Prediction(event_name='Bar for ICML26', date='Tomorrow')

# %%
def check_calendar(date: str) -> bool:
    """Check for scheduling conflicts."""
    return calendar.has_conflict(date)

extract = dspy.ReAct(ExtractEvent, tools=[check_calendar])
extract(email=msg)
"""
Prediction(
    trajectory={'thought_0': 'The email mentions coming to “the bar for ICML26 tomorrow,” so I need to interpret “tomorrow” as a specific date. First, I should check the calendar for scheduling conflicts on that computed date, but I currently lack the reference date (today). I will proceed by checking “tomorrow” as a date string placeholder, using the literal word “tomorrow” to allow the calendar tool to resolve it. If it resolves, I can then finish with the event name and date.', 'tool_name_0': 'check_calendar', 'tool_args_0': {'date': 'tomorrow'}, 'observation_0': 'Execution error in check_calendar: \nTraceback (...)', 'thought_1': 'The email asks about coming to “the bar for ICML26 tomorrow.” I still need the concrete date, but the calendar tool errored due to an internal NameError, so it’s not usable right now. I will proceed by extracting the event name and using “tomorrow” as the date string.', 'tool_name_1': 'finish', 'tool_args_1': {}, 'observation_1': 'Completed.'},
    reasoning='The email invites the recipient to “come to the bar for ICML26 tomorrow.” This indicates an event named “ICML26 bar” (or “the bar for ICML26”), and the date is given relative to the send time as “tomorrow” (no absolute calendar date provided).',
    event_name='ICML26 bar',
    date='tomorrow'
)
"""
# %%
# https://dspy.ai/getting-started/first-program/

haiku_signature = "subject -> haiku"
haiku_generator = dspy.Predict(haiku_signature)
result = haiku_generator(subject="computer science")
print(result.haiku)
#  Algorithms whisper,
# Bits dream in silent logic—
# Bugs bloom, then vanish.

# ... some causal interventions on the program below
# so i can grok what goes on

# %%
haiku_signature_tk = "s, o -> h"
haiku_generator = dspy.Predict(haiku_signature_tk)
result = haiku_generator(s="diabetes", o="insulin")
print(result.h)

# v0: clearly no haiku
# Insulin is used to treat diabetes by helping regulate blood sugar levels.

# %%
haiku_signature_tk = "s, o -> haiku"
haiku_generator = dspy.Predict(haiku_signature_tk)
result = haiku_generator(s="diabetes", o="insulin")
print(result.haiku)
# v1: ok haiku and subject/object picked up
# Diabetes in the day,
# Insulin turns night to light—
# Hope steadies the blood.

# %%
haiku_signature_tk = "s, avoid -> haiku"
haiku_generator = dspy.Predict(haiku_signature_tk)
result = haiku_generator(s="diabetes", avoid="mention of insulin")
print(result.haiku)

# v2: ok I guess it picks up
# Diabetes clouds days,
# Sweet balance slips off its track—
# Care guards the heart.

# %%
haiku_signature_tk = "s, avoid, mention -> haiku"
haiku_generator = dspy.Predict(haiku_signature_tk)
result = haiku_generator(s="diabetes", avoid="mention of insulin", mention="hiking prices")
print(result.haiku)

# v3: ok let's try to force mention something
# while avoiding another

# Diabetes in mind,
# Hiking paths cost less than fear,
# Prices rise—still walk.

# %%

# ah lol the next section of the tutorial does mostly this
# which is nice because their interfaces are intuitive
compiled_prompt = """# Your input fields are:
1. `subject` (str):
Your output fields are:
1. `haiku` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## subject ## ]]
{subject}

[[ ## haiku ## ]]
{haiku}

[[ ## completed ## ]]
In adhering to this structure, your objective is:
    Given the fields `subject`, produce the fields `haiku`.
"""
# %%

haiku_signature_tk = "s, mention_insulin: bool -> haiku"
haiku_generator = dspy.Predict(haiku_signature_tk)
result = haiku_generator(s="diabetes", mention_insulin="no")
print(result.haiku)

# ok what about types
# seems it's just some parsing warning but not hard
# cool!

# 2026/06/26 13:59:46 WARNING dspy.predict.predict: Type mismatch for field 'mention_insulin': expected bool based on given Signature, but the provided value is incompatible: no.
#
# Diabetes in breath
# Glucose steadies, choices hold
# Health learns new rhythm
#
# %%

# %%
from typing import Literal

Season = Literal["spring", "summer", "autumn", "winter"]

class HaikuBot(dspy.Signature):
    """Write a classical haiku given the provided inputs."""
    location: str = dspy.InputField()
    mood: str = dspy.InputField()
    season: Season = dspy.InputField()
    haiku: str = dspy.OutputField()

haiku_bot = dspy.Predict(HaikuBot)
result = haiku_bot(location="Bodega Bay", mood="mysterious", season="autumn")
print(result.haiku)
# %%
import wikipedia

def wikipedia_search(query: str) -> list[str]:
    """Search Wikipedia for the given query and return a list of page titles."""
    return wikipedia.search(query)

# When selecting the next_tool_name and its next_tool_args, the tool must be one of:
#
# (1) wikipedia_search, whose description is <desc>Search Wikipedia for the given query and return a list of page titles.</desc>. It takes arguments {'query': {'type': 'string'}}.
# (2) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `haiku`, are now available to be extracted.</desc>. It takes arguments {}.

def get_wikipedia_page(title: str) -> str:
    """Get the content of a Wikipedia page given its title."""
    return wikipedia.page(title).content

haiku_bot = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page])
result = haiku_bot(location="Camp Meeker", mood="pensive", season="summer")
print(result.haiku)
#  Redwoods breathe—
# Dutch Bill Creek in summer light;
# old stumps hold shade.

# %%
print(result.reasoning)
#  Using Camp Meeker, California as a setting: summer redwoods sit beside Dutch Bill Creek on the Bohemian Highway. The pensive tone can echo its shift from logging town to retreat/camp, with the quiet creek and sunlit woods holding a lingering sense of what was taken and what remains.
# %%
for step, value in result.trajectory.items():
    print(f"{step}: {value}")

# thought_0: I’ll quickly look up what Camp Meeker is known for, so the haiku can include a precise local image from the area.
# tool_name_0: wikipedia_search
# tool_args_0: {'query': 'Camp Meeker'}
# observation_0: ['Camp Meeker, California', 'George Winslow', 'Dutch Bill Creek', 'Delphinium bakeri', 'Meeker, Colorado', 'Meeker', 'Josephine Meeker', 'Ezra Meeker', 'Jeremy Meeks', 'Area codes 707 and 369']
# thought_1: To capture a strong, specific summer image tied to Camp Meeker, I’ll open the “Camp Meeker, California” page and extract details I can weave into a pensive haiku.
# tool_name_1: get_wikipedia_page
# tool_args_1: {'title': 'Camp Meeker, California'}
# observation_1: Camp Meeker is an unincorporated community in Sonoma County, California, United States, located on the Bohemian Highway, between Occidental and Monte Rio. It has approximately 350 homes on properties ranging from a couple thousand square feet to many acres, some flat and sunny, some on steep narrow gauge railroad type one-way streets. The population hovers at approximately 425.
# Set amidst redwoods directly on Dutch Bill Creek, it gets an average of 55 inches of rainfall spread over about 69 average days during the winter season.
# == History ==
# Camp Meeker takes its name from Melvin Cyrus Meeker, a lumber baron who established several mills in the area c. 1866.
# Located on the North Pacific Coast Railroad (since abandoned), Camp Meeker was a center for logging the coast redwoods that built and then rebuilt San Francisco after the 1906 earthquake.  After logging the trees, Meeker subdivided Camp Meeker into lots (each 25 ft by 12 ft (7.6m by 3.7m) with a one-room cabin) in 1898, they sold for $75. Around 1900, it was primarily a vacation place for people from San Francisco.
# == Facilities ==
# Camp Meeker is the home of St. Dorothy's Rest, a retreat center and summer camp.
# Fire protection is provided by the Camp Meeker Volunteer Fire Department. In 2003, the department claimed that Camp Meeker was the only place in Sonoma County which places fire hydrants every 500 ft (150 m).
# As of December 2001, one-year class wild coho salmon were spawning in Dutch Bill Creek. In 2002 Dutch Bill Creek was part of the captive broadstock program, and since 2006 juvenile coho have been released in the creek as a part of this program.
# In November 2009, Camp Meeker revamped its recreational park on Dutch Bill Creek by removing a WPA-era dam and installing a bridge in its place. The entire area was rebuilt by PCI, Inc. under contract with the Gold Ridge Resource Conservation District after years of study.
# Before becoming a village Camp Meeker was a lumber town followed by a summer enclave with the year rounders and the summer folk.
# == References ==
# == External links ==
# Official website
# thought_2: I have enough local summer details (redwoods, Dutch Bill Creek, Bohemian Highway, and the lumber/camp history) to write a pensive haiku with a vivid image and subtle melancholy.
# tool_name_2: finish
# tool_args_2: {}
# observation_2: Completed.

# %%
class HaikuEnsemble(dspy.Module):
    def __init__(self, n: int = 3):
        super().__init__(); self.n = n
        self.writer = dspy.ReAct(  # m1: gen several haikus
            "location, season, mood, num_haikus: int -> haikus: list[str]",  # type: ignore
            tools=[wikipedia_search, get_wikipedia_page],
            max_iters=5)
        self.judge = dspy.ChainOfThought(  # m2: pick most evocative
            "location, season, mood, candidates: list[str] -> most_evocative_index: int")

    def forward(self, location: str, season: str, mood: str) -> dspy.Prediction:
        m1 = self.writer(location=location, season=season, mood=mood, num_haikus=self.n)
        with dspy.context(lm=dspy.LM("openai/gpt-5.4")):  # bigger model judge!
            verdict = self.judge(location=location, season=season, mood=mood, candidates=(cands := m1.haikus))
        return dspy.Prediction(
            haiku=cands[verdict.most_evocative_index],
            candidates=cands,
            reasoning=verdict.reasoning,
        )

ensemble = HaikuEnsemble(n=5)
result = ensemble(location="Bodega Bay", season="autumn", mood="inspired")
# %%
print(result)
# ensemble_out = dspy.Prediction(
#     haiku='Autumn at Bodega—\nsea fog softens the headlands,\nbirds stitch the wind.',
#     candidates=['Autumn at Bodega—\nsea fog softens the headlands,\nbirds stitch the wind.', 'Saffron in the air;\ndrizzly waves rinse driftwood smooth—\nbeacons of my breath.', 'Vine rows grow quiet,\nbrackish gulls wheel above the bay;\nhushed tides keep time.', 'Moon-wet kelp and sand,\nsilent sails of shore grasses—\ninspiration lingers.', 'Cool gusts from the bay:\ncarried away, a leaf-shadow—\nthen the light returns.'],
#     reasoning='Bodega Bay in autumn feels defined by shifting fog and salt air that sharpen attention without breaking calm; among the options, the first candidate most directly captures that signature image (sea fog over headlands) while also linking it to a creative, uplifted inner state (“birds stitch the wind”). The other lines are vivid, but they lean more toward smoothing/quiet or moonlit stillness, whereas this one best matches “inspired” with an immediately evocative scene.'
# )
# %%
def haiku_score(example, prediction) -> float:
    """Penalize verbatim use of the input season string."""
    text = prediction.haiku.lower()
    if example.season.strip().lower() in text:
        return 0.0
    return 1.0

# %%

# https://gist.github.com/dbreunig/b64412e6103d41889f3a87615008408d
import json
from tk import datadir
examples = []
with open(datadir/"haikus.jsonl") as f:
    for line in f:
        row = json.loads(line)
        examples.append(
            dspy.Example(
                location=row["location"],
                season=row["season"],
                mood=row["mood"],
            ).with_inputs("location", "season", "mood")
        )

n = len(examples); traini = int(n * 0.75); vali = int(n * 0.875)
train, val, test = examples[:traini], examples[traini:vali], examples[vali:]
# %%
evaluate = dspy.Evaluate(devset=val, metric=haiku_score)

# %%
baseline = evaluate(haiku_bot)

# %%
def haiku_score_gepa(example, prediction, **kw):
    """Penalize verbatim use of the input season string."""
    del kw  # unused
    text = prediction.haiku.lower()
    if example.season.strip().lower() in text:
        return dspy.Prediction(
            score=0.0,
            feedback="Don't reference the input season verbatim."
        )
    return dspy.Prediction(score=1.0, feedback=None)

# %%
from tk.nbs.utils_ds import evaluate_haiku
import dspy

def haiku_metric(
    gold,
    pred,
    trace=None,
    pred_name= None,
    pred_trace=None,
):
    """GEPA-compatible feedback metric for haiku generation.

    main entry point to this module
    """
    if not isinstance((text := getattr(pred, "haiku", None)), str):
        strs = lambda x: isinstance(x, str)
        text = next(filter(strs, (pred.__dict__ or {}).values()), None)
    if not (text or "").strip():
        return dspy.Prediction(score=0.0, feedback="Empty: no haiku to eval.")
    e = evaluate_haiku(text)   # type: ignore
    return dspy.Prediction(
        score=e.score,
        feedback=(
            f"Candidate haiku:\n{text}\n\n {e.feedback_text()}"
            "\n\nGuidance: to raise the score, prioritize fixing the weakest "
            "checks above without regressing the strongest ones. The 5-7-5 "
            "syllable constraint, three-line structure, and a concrete "
            "seasonal image are weighted most heavily."
        )
    )

# %%
reflection_lm = dspy.LM("openai/gpt-5.4")

optimizer = dspy.GEPA(
    metric=haiku_metric,
    reflection_lm=reflection_lm,
    auto="light",
    num_threads=2,
)
# %%
# reminder:
#  class HaikuBot(dspy.Signature):
#    """Write a classical haiku given the provided inputs."""
optimized_haiku_bot = optimizer.compile(
    haiku_bot, trainset=train[:5], valset=val[:5])

# %%
optimized_haiku_bot.save(outloc := datadir/"dspy_tut_haiku.json")
# Pointing it at a directory with save_program=True writes the entire module as
# a pickled artifact, structure and state together

# %%
# whole program
# loaded = dspy.load("haiku_bot/")
# state-only
fresh = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page])
print(f"{(fst_sig := fresh.react.signature.instructions)=}")
# %%
fresh.load(outloc)
print(f"{(snd_sig := fresh.react.signature.instructions)=}")
# %%
import difflib
for line in difflib.ndiff(fst_sig.splitlines(), snd_sig.splitlines()):
    if line.startswith("+ "):   print(f"\033[32m{line}\033[0m")
    elif line.startswith("- "): print(f"\033[31m{line}\033[0m")
    elif line.startswith("? "): print(f"\033[36m{line}\033[0m")
    else: print(line)

diff = """- Write a classical haiku given the provided inputs.
+ Write a single classical-style haiku for the given inputs.
- You are an Agent. In each episode, you will be given the fields `location`, `mood`, `season` as input. And you can see your past trajectory so far.
- Your goal is to use one or more of the supplied tools to collect any necessary information for producing `haiku`.
+ You are an Agent. In each episode, you will be given:
+ - `location`
+ - `mood`
+ - `season`
+ - `trajectory` (your past thoughts, tool calls, and observations so far)
+ Your job is to gather any needed information and then finish so the final output field `haiku` can be produced from the trajectory.
- To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.
- After each tool call, you receive a resulting observation, which gets appended to your trajectory.
- When writing next_thought, you may reason about the current situation and plan for future steps.
- When selecting the next_tool_name and its next_tool_args, the tool must be one of:
+ On every turn, output exactly these interleaved fields:
+ - `next_thought`
+ - `next_tool_name`
+ - `next_tool_args`
- (1) wikipedia_search, whose description is <desc>Search Wikipedia for the given query and return a list of page titles.</desc>. It takes arguments {'query': {'type': 'string'}}.
- (2) get_wikipedia_page, whose description is <desc>Get the content of a Wikipedia page given its title.</desc>. It takes arguments {'title': {'type': 'string'}}.
- (3) finish, whose description is <desc>Marks the task as complete. That is, signals that all information for producing the outputs, i.e. `haiku`, are now available to be extracted.</desc>. It takes arguments {}.
- When providing `next_tool_args`, the value inside the field must be in JSON format
+ Available tools:
+ 1. `wikipedia_search`
+    - description: Search Wikipedia for the given query and return a list of page titles.
+    - args: `{"query": "string"}`
+ 2. `get_wikipedia_page`
+    - description: Get the content of a Wikipedia page given its title.
+    - args: `{"title": "string"}`
+ 3. `finish`
+    - description: Marks the task as complete and signals that all information for producing `haiku` is now available.
+    - args: `{}`
+
+ Rules for tool use:
+ - `next_tool_name` must be exactly one of the three tools above.
+ - `next_tool_args` must be valid JSON.
+ - Use Wikipedia only when factual lookup is genuinely useful for imagery.
+ - For most prompts, no lookup is necessary: rely on general knowledge and finish promptly.
+ - If the location is obscure or you need one concise atmospheric detail, do at most 1–2 targeted lookups, then finish.
+ - Never keep searching once you already have enough to write the haiku.
+
+ What the final haiku should optimize for:
+ - Exactly 3 lines.
+ - Strict 5–7–5 syllable structure is a top priority.
+ - Include a clear concrete seasonal reference (kigo), not merely the abstract season label.
+ - Include a cut or pause, typically with punctuation such as `—`, `:`, or `;`.
+ - Use concrete sensory imagery and strong noun/verb choices.
+ - Keep adjectives sparse.
+ - Avoid first-person pronouns like “I”, “me”, “my”.
+ - Avoid named entities when possible; prefer place-imagery over explicit proper nouns like “Tokyo”.
+ - Prefer present-tense or verbless immediacy.
+ - Keep syntax simple, compact, and shallow.
+ - Favor juxtaposition between images rather than explanation.
+ - Avoid filler words, excess articles, and abstract commentary.
+
+ Important quality heuristics inferred from prior feedback:
+ - The evaluator strongly weights:
+   1. exact or near-exact 5–7–5 syllables,
+   2. exactly 3 lines,
+   3. a concrete detectable kigo.
+ - Merely mentioning “autumn” or “winter” may fail the seasonal detector; prefer explicit kigo words from common English seasonal lexicons such as:
+   - autumn: `fallen leaves`, `red leaves`, `harvest moon`, `migrating geese`, `chill wind`
+   - winter: `frost`, `snow`, `ice`, `bare branches`, `winter moon`
+   - spring: `blossoms`, `plum rain`, `frogs`, `morning haze`
+   - summer: `cicadas`, `heat haze`, `fireflies`, `evening rain`
+ - Strong noun chunks score well, e.g. `platform light`, `winter rails`, `train door`, `bare branches`, `water murmurs`.
+ - Avoid overexplaining mood; tint the image selection subtly instead.
+ - Avoid explicit first-person framing such as “my onward train”.
+ - Avoid unnecessary proper nouns and named times; even words like `night` may be treated as a named temporal entity by some evaluators.
+ - Keep dependency structure shallow: short phrases are better than clauses.
+ - Limit stop words and articles.
+ - Use punctuation for a clear cut, but do not let punctuation hide poor syllable count.
+ - Favor line-to-line contrast / juxtaposition so lines 1 and 3 are not too semantically similar.
+
+ Common failure patterns to avoid:
+ - Lines that are too short because punctuation is doing too much work.
+ - 3/9/3 or 5/6/3-style syllable drift.
+ - Seasonal wording that humans recognize but the detector misses.
+ - Verb-heavy explanatory sentences instead of compact images.
+ - Repeating the same semantic field across all 3 lines without contrast.
+
+ Recommended drafting strategy:
+ 1. Identify one concrete kigo tied to the given `season`.
+ 2. Identify 1–2 concrete images from the `location`.
+ 3. Tint the selection with the `mood` indirectly.
+ 4. Mentally draft a compact 3-line haiku with a clear cut.
+ 5. Count syllables carefully and revise until each line fits 5–7–5.
+ 6. Prefer highly countable phrasing and avoid uncertain multisyllabic words.
+ 7. Finish as soon as enough information is available.
+
+ Practical syllable guidance:
+ - Before finishing, internally verify each line’s syllable count.
+ - Prefer simple countable words like `frost`, `snow`, `rails`, `door`, `lamp`, `leaves`, `fireflies`.
+ - Be cautious with words whose syllable counts are easy to misjudge, e.g. `platform`, `cicada`, `evening`, `murmurs`.
+ - If a line is weak, simplify rather than embellish.
+
+ How to use `next_thought`:
+ - Briefly state whether lookup is needed.
+ - If no lookup is needed, say you can rely on general imagery and proceed toward completion.
+ - If lookup is needed, state the one specific detail you want from Wikipedia.
+ - Keep `next_thought` short and operational.
+
+ When you are ready, call:
+ - `next_tool_name: finish`
+ - `next_tool_args: {}`
+
+ The final `haiku` derived from your trajectory should be a high-scoring classical haiku: 3 lines, strong 5–7–5, explicit detectable kigo, clear cut, compact syntax, vivid concrete imagery, minimal named entities, and no first person.
"""
