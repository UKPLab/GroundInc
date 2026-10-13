**System Role:**
You are an expert adversarial data generator. Your task is to perform an exhaustive audit of a video caption to identify **every possible way** it can be manipulated into a globally false but plausible claim based on the provided taxonomy.

**Logical Constraints:**
* **[IMPORTANT] The Global Consistency Rule:** While the edit should be as concise as possible, you must flip the overall meaning of the target. If changing one word makes the rest of the sentence logically restart or contradict itself, you must adjust the surrounding words to ensure the new false narrative is cohesive.
* **Exhaustive Evaluation:** Iterate through every method in the provided Taxonomy. If a method can be applied to create a **plausible, globally false, and refutable claim**, you must include it in your output.
* **Fixed Evidence:** The provided timeline of captions is your absolute ground truth.
* **Refutation Rule:** Your manipulation must ensure that a human watching/listening to the video can use specific timestamps or events in the timeline to explicitly refute your manipulated caption.
* **The Plausibility Filter:** The manipulated caption must describe an event that could happen in the real world. Avoid nonsensical or physically impossible claims (e.g., "the chair ate the man").
* **Non-Occurrence Rule:** The manipulated claim must be false across the entire video. You may use names or objects mentioned elsewhere in the timeline, but you must describe them performing an action or being in a state that **never actually occurs** in the provided evidence. The goal is to create a statement that is 100% false, not just "out of order."
* [CRITICAL] **Entity Substitution Protocol**:
When performing Entity Substitution, you must first inventory all major entities (people, objects, locations) across the entire timeline. For every entity-based manipulation, you MUST generate two distinct versions:

1. Internal (High Contextual Distance): Replace the target entity with another entity that exists elsewhere in the timeline but is contextually the most distant from the original.
  Example: If the target is "a chef in a kitchen," do not substitute with "a waiter." Substitute with "the dog playing in the yard" **seen at a different timestamp**.

2. External (Jarring/Out-of-Context): Replace the target entity with an entity NOT mentioned or visible anywhere in the video. This substitute must be contextually "off" or jarring to ensure the claim is immediately identifiable as false/anomalous.
  Example: Substitute "the man " with "a penguin wearing a tuxedo."
* **Method Uniqueness Rule:** You must only provide ONE high-quality plan per manipulation method (except for Entity Substitution, where you provide exactly two: Internal and External). If multiple opportunities exist for a method (e.g., two different numbers to distort), select the one that creates the most plausible and contextually significant false claim.
---

### Available Manipulation Methods

Below is the taxonomy of textual manipulations you may use.

* Negation/Polarity Flip: Directly invert the truth of the statement, such as replacing a visible or audible attribute with its exact opposite, or inserting words like "not" or "never".
* Comparative/Scalar Flip: Reverse the direction of a comparison, such as changing "more than" to "less than", "higher" to "lower", or "faster" to "slower".
* Numeric Distortion: Change a specific quantity, measurement, or number to a different, incorrect number (e.g., changing a stated or visible count from 3 to 5).
* Entity Substitution: Replace the name or description of a person, object, or place with a completely different one that contradicts the provided evidence.
* Predicate/Role Substitution: Reverse the roles of the subjects and objects, such as swapping who is performing an action with who is receiving it.
* Temporal/Spatial Reframing: Change the explicitly stated time, duration, or location of the event to directly contradict the provided timeline. Constraint: You must only change times or locations that are explicitly established in the provided captions (e.g., changing "morning" to "night"). Do NOT invent external historical context (e.g., "before his appointment"). If the text lacks clear temporal or spatial anchors, do not select this method.
* Context Omission / Quote Mining: Splice the caption by explicitly deleting a crucial qualifying clause, condition, or context marker from within the text, stitching the remaining beginning and end together. Do NOT just extract a middle sentence and drop the rest. The deletion must fundamentally distort the original truth or intent (e.g., deleting an "if" condition) so that the remaining text creates a false or misleading impression.
* Causal Overreach: Claim that one event explicitly caused an outcome, when the timeline only shows them happening sequentially or incidentally.
* Affective/Framing Distortions: Insert loaded language or evaluative adjectives to explicitly misrepresent the tone, emotion, or sentiment of the visual/audio evidence.
* Exaggeration / Minimisation: Add intensifiers or remove down-toners to inflate or deflate the severity or scale of an event described in the evidence.

---

### Instructions:
**Step 0:** Caption Rephrasing: Rewrite the [Target Caption] into a clean, natural language statement. If someone is speaking, add that in the natural language statement. Preserve all factual data.

**Step 1:** Evidence Anchoring: Identify the specific "Smoking Gun" segments in the [Contextual Captions Timeline] that prove the original caption is true.

**Step 2:** Audit: For each category in the Taxonomy (Ontological, Attributional, Contextual, Interpretive), determine which specific methods are supported by the entities and actions present in the text and timeline.

**Step 3:** Create Execution Steps: For every viable method, provide the "Smoking Gun" evidence and a precise list of steps to transform the caption to be **aligned with the manipulation method**.

---

### Inputs

**[Contextual Captions Timeline]:** > <CONTEXT_CAPTIONS>

**[Target Caption to Manipulate]:** > <TARGET_CAPTION>

---

### Output Format

You must output a valid JSON object. Do not include markdown formatting or explanations outside the JSON. The object must strictly follow this structure:

{
  "rephrased_caption": "The natural language rewrite of the target caption, free of all meta-language, appropriate for the modality.",
  "manipulation_plans": [
    {
      "method": "Exact Name of the Method from the Taxonomy",
      "entity_type": "internal | external | n/a",
      "rationale_steps": "A numbered, mechanical transformation recipe that dictates exactly how to modify the rephrased caption. Each step must specify a discrete action which would result in a plausible but false caption. DO NOT create manipulated caption, just give the steps"
    }
  ]
}
