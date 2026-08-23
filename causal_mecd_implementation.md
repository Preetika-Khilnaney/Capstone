I want you to IMPLEMENT TRACK 2 — THE CAUSAL ENGINE — exactly as specified in my project architecture.

IMPORTANT:

The current project already has the perception/data pipeline:

VIDEO
→ YOLO object detection
→ MOG2 motion detection
→ BoT-SORT tracking
→ persistent Object_ID
→ Homography
→ world-space coordinates
→ kinematic data / CSV

assume that The CAUSAL ENGINE DOES NOT EXIST YET. you may delete all the causal files

You must implement Track 2 from scratch.

Do NOT rebuild Track 1 unless a required output is missing or incorrect.

The causal engine must consume the existing kinematic data and produce:

Kinematic Time Series
→ Causal Discovery
→ Temporal Event Reasoning
→ Event Causal Graph
→ Exact Event Onset
→ Causal Explanation

============================================================
1. FOLLOW THE EXISTING PROJECT DOCUMENT
============================================================

The project specification defines Track 2 as:

"Kinematics → Causal Graph"

The core problem is:

Given kinematic time-series from multiple vehicles:

WHAT CAUSED WHAT?

The system must answer questions such as:

- Did Vehicle A's deceleration cause Vehicle B to brake?
- Did separation distance cause relative velocity change?
- Which vehicle initiated the interaction chain?
- What event caused the subsequent event?
- What was the earliest meaningful point of the incident?

Implement this as an actual causal-analysis system, not merely a visualization.

============================================================
2. EXISTING INPUT DATA
============================================================

The current Track 1 produces data similar to:

Event_ID
Timestamp
Frame_ID
Object_ID
Class
BBox_X1
BBox_Y1
BBox_X2
BBox_Y2
Pos_X_m
Pos_Y_m
Velocity_mps

Use the existing schema if present.

However, BEFORE causal analysis, extend/verify the data so that every tracked entity has:

timestamp
frame_id
object_id
class

world_x
world_y

vx
vy
speed

ax
ay
acceleration

heading

detection_confidence
tracking_confidence
position_source
velocity_source

Do not invent missing physical quantities.

============================================================
3. IMPORTANT: FIX VELOCITY BEFORE CAUSAL ANALYSIS
============================================================

The causal engine depends heavily on velocity and acceleration.

The existing method currently calculates velocity using:

sqrt(dx² + dy²) / dt

after converting the bottom-center bounding-box point into world coordinates.

Keep the existing Homography-based world coordinates, but improve the kinematic calculation if necessary.

Use:

vx = Δx / Δt
vy = Δy / Δt

speed = sqrt(vx² + vy²)

acceleration:

ax = Δvx / Δt
ay = Δvy / Δt

acceleration magnitude:

a = sqrt(ax² + ay²)

heading:

atan2(vy, vx)

Use the ACTUAL timestamp difference between frames.

Do not blindly assume:

dt = 0.1

unless the actual video timing proves this.

This is critical for causal analysis.

============================================================
4. UNIQUE ENTITY IDENTITY
============================================================

Every physical entity must retain a unique Object_ID throughout the video.

Use the existing BoT-SORT tracking IDs.

Example:

V_01 = Car
V_02 = Motorcycle
V_03 = Car

The causal engine must reason about:

V_01_Velocity
V_02_Velocity

rather than treating all cars as one variable.

Do not create a new identity every time YOLO detects an object.

Handle short occlusions using the existing tracker.

Track history must remain associated with the same Object_ID.

============================================================
5. DO NOT USE OBJECT_ID AS A CAUSAL VARIABLE
============================================================

Object_ID is an identity.

It must NEVER be interpreted as a causal feature.

Correct:

V_01_speed
V_01_acceleration
V_02_speed
V_02_acceleration

Incorrect:

Object_ID → Accident

============================================================
6. CREATE ENTITY TIME SERIES
============================================================

For every Object_ID create a clean time series.

Example:

V_01:

timestamp
world_x
world_y
vx
vy
speed
ax
ay
acceleration
heading

V_02:

timestamp
world_x
world_y
vx
vy
speed
ax
ay
acceleration
heading

Maintain temporal ordering.

============================================================
7. CREATE PAIRWISE INTERACTION FEATURES
============================================================

This is REQUIRED.

Causal reasoning should not only analyze individual vehicles.

For every relevant pair:

Vehicle A
Vehicle B

calculate:

distance
distance_rate
relative_x
relative_y
relative_vx
relative_vy
relative_speed
closing_speed
relative_acceleration
heading_difference
trajectory_difference
overlap/contact indicator

Example:

V_01 ↔ V_02

distance(t)
closing_speed(t)
relative_velocity(t)

These features allow the system to determine whether one entity's behavior influenced another.

============================================================
8. DETERMINE RELEVANT VEHICLE PAIRS
============================================================

Do not blindly create pairwise features for every object.

Prioritize pairs that:

- become spatially close
- have intersecting trajectories
- have decreasing distance
- have high closing speed
- occupy the same interaction region
- show abnormal simultaneous motion changes

This makes the causal engine scalable.

============================================================
9. BUILD THE STATISTICAL CAUSAL ENGINE
============================================================

Implement the following methods as specified in the project architecture:

1. PCMCI+
2. LiNGAM
3. GES

Also implement:

4. Causal Forest

If CycleNet is genuinely available and installable, implement it.

If it is NOT actually available:

DO NOT fake CycleNet.

Instead clearly mark it:

"CycleNet unavailable"

and continue with the available methods.

============================================================
10. PCMCI+ — PRIMARY TEMPORAL CAUSAL METHOD
============================================================

Use Tigramite PCMCI+.

PCMCI+ is especially important because traffic data is time-series data.

The engine should test relationships such as:

V_01_speed(t-1)
→
V_02_speed(t)

or:

V_01_deceleration(t-2)
→
V_02_deceleration(t)

or:

distance_V01_V02(t-1)
→
relative_velocity_V01_V02(t)

Use appropriate temporal lags.

Do NOT only calculate correlation.

The purpose is causal discovery with temporal conditioning.

Store:

source
target
lag
strength
p_value
method

============================================================
11. LAG CONFIGURATION
============================================================

Make lag configuration configurable.

Example:

tau_min = 1
tau_max = 10

Do NOT hardcode tau_max=3 without considering the video's FPS and temporal resolution.

Convert frame lags into meaningful time where possible.

For example:

lag = 5 frames
FPS = 25

means approximately:

0.20 seconds

Store both:

lag_frames
lag_seconds

============================================================
12. LiNGAM
============================================================

Implement DirectLiNGAM or the most appropriate available LiNGAM method.

Use it primarily to provide additional directional evidence.

Input:

standardized causal variables

Output:

causal adjacency matrix.

Convert the result into:

source
target
strength
method = "LiNGAM"

Do not claim LiNGAM proves physical causality.

Treat it as causal-discovery evidence under its assumptions.

============================================================
13. GES
============================================================

Implement GES using causal-learn if available.

Use an appropriate score such as BIC.

Extract:

source
target
score
method

Normalize its output so it can be compared with other methods.

============================================================
14. CAUSAL FOREST
============================================================

Use Causal Forest only where a valid treatment/outcome formulation exists.

DO NOT randomly run a causal forest on every column.

For example, a meaningful question could be:

Treatment:
V_01 sudden deceleration

Outcome:
V_02 subsequent deceleration

Control variables:

distance
relative speed
initial speed
heading difference

Use causal forest to estimate heterogeneous treatment effects where appropriate.

Store:

treatment
outcome
effect
confidence
method

If a causal forest formulation is not statistically valid for a particular event, skip it and explain why.

============================================================
15. CYCLIC RELATIONSHIPS
============================================================

Traffic can contain feedback-like interactions.

Example:

Vehicle A changes speed
→ Vehicle B reacts
→ Vehicle A reacts again

Do not force everything into a simple static DAG.

If CycleNet is available, use it.

Otherwise represent the temporal feedback explicitly:

A(t)
→ B(t+Δt)
→ A(t+2Δt)

rather than falsely creating:

A ↔ B

without temporal information.

============================================================
16. MULTI-METHOD CONSENSUS
============================================================

This is a major part of the causal engine.

For every candidate causal edge:

A → B

collect results from:

PCMCI+
LiNGAM
GES
CycleNet
Causal Forest

Example:

PCMCI+     YES
LiNGAM     YES
GES        YES
CycleNet   unavailable
Forest     YES

Create:

method_count
available_methods
support_count
support_ratio
average_strength
final_confidence

Example:

support:
4 / 4 available methods

confidence:
0.91

Do NOT count unavailable methods as failures.

============================================================
17. DO NOT SIMPLY AVERAGE DIFFERENT SCORES
============================================================

Different causal algorithms produce different types of scores.

Normalize scores before combining.

Maintain both:

raw_method_result

and:

normalized_consensus_result

The final graph must preserve the evidence from each method.

============================================================
18. CAUSAL GRAPH — STATISTICAL GRAPH
============================================================

Create the first graph:

STATISTICAL CAUSAL GRAPH

Example:

V_01_Deceleration
        ↓
V_02_Deceleration

or:

Distance_V01_V02
        ↓
Relative_Velocity_V01_V02

Each edge must contain:

source
target
lag_frames
lag_seconds
strength
p_value if available
methods
support_count
confidence

============================================================
19. DO NOT STOP AT THE STATISTICAL GRAPH
============================================================

This is extremely important.

The statistical graph alone does NOT solve my project's main problem.

The system must additionally construct an:

EVENT-LEVEL TEMPORAL CAUSAL GRAPH.

The statistical graph explains relationships between variables.

The event graph explains:

WHAT HAPPENED
→
WHAT HAPPENED NEXT
→
WHY THE NEXT EVENT OCCURRED

============================================================
20. TEMPORAL EVENT EXTRACTION
============================================================

From the kinematic time series, convert important changes into temporal events.

Examples:

V_01:

Normal movement
→
Sudden braking

V_02:

Normal movement
→
Sudden braking

Pair:

Distance decreasing
→
Contact

V_02:

Sudden velocity change
→
Trajectory deviation
→
Fall

Each event must have:

event_id
object_ids
event_type
start_frame
end_frame
start_timestamp
end_timestamp
features
confidence

============================================================
21. EVENT TYPES MUST BE GENERIC
============================================================

Support:

SUDDEN_BRAKING
SUDDEN_ACCELERATION
SUDDEN_STOP
SUDDEN_DIRECTION_CHANGE
SWERVE
LANE_DEVIATION
APPROACHING
CLOSING_DISTANCE
CONTACT
NEAR_COLLISION
COLLISION
FALL
TRAJECTORY_DEVIATION
ABNORMAL_MOTION

Do NOT hardcode only:

CAR_BIKE_COLLISION.

The engine must derive event types from the observed data.

============================================================
22. EVENT CAUSALITY
============================================================

For two events:

Event A
Event B

A can only be considered a causal candidate for B if:

1. A occurs before B
2. A contains information that helps explain/predict B
3. there is supporting physical/kinematic evidence
4. the relationship is not explained only by a common simultaneous change

Use temporal/Granger-style reasoning as an additional criterion.

For example:

V_01 sudden braking
at 12.4s

followed by:

V_02 sudden braking
at 12.8s

Test whether V_01's braking provides predictive information for V_02's braking beyond V_02's own previous behavior.

If supported:

V_01 braking
        ↓
V_02 braking

If not:

do NOT label it causal merely because it happened earlier.

============================================================
23. CRITICAL: CAUSALITY VS TEMPORAL ORDER
============================================================

Never assume:

A happened before B
therefore
A caused B.

The graph must distinguish:

PRECEDES

from:

CAUSES

from:

SUPPORTS

Example:

Rain
→
Wet road

may be causal.

Car A passes Car B
→
Car B brakes

may only be temporal unless supported by interaction evidence.

============================================================
24. ACTUAL ACCIDENT ONSET
============================================================

THIS IS THE MOST IMPORTANT FIX.

The current system has a problem:

Actual collision:
approximately 00:13

but event is displayed:
approximately 00:20.

The reason is likely that the system is detecting a later consequence/confirmation rather than the actual initiating event.

Implement a dedicated:

EventOnsetLocalizer

============================================================
25. EVENT ONSET LOCALIZATION
============================================================

For every detected event:

1. Identify confirmation time.
2. Search backward through the event history.
3. Identify precursor events.
4. Identify interaction changes.
5. Identify physical transition.
6. Determine the earliest frame where the defining event actually occurs.

For a collision:

Normal movement
↓
Approaching
↓
Distance decreasing
↓
Closing speed increasing
↓
CONTACT
↓
Sudden relative velocity change
↓
Trajectory deviation
↓
Fall
↓
Confirmation

The actual collision onset should correspond to the earliest physically supported:

CONTACT / impact transition

NOT:

fall
NOT:
stopped position
NOT:
large MOG2 motion after the crash
NOT:
final event confirmation.

============================================================
26. DO NOT USE FIXED OFFSET
============================================================

ABSOLUTELY DO NOT implement:

detected_time - 7 seconds

or:

detected_time - 5 seconds

or:

event_time = 13 seconds.

The onset must be calculated dynamically from the video-derived evidence.

============================================================
27. EVENT ONSET AND CONFIRMATION MUST BOTH BE STORED
============================================================

Every event must contain:

event_onset:
frame
timestamp

confirmation:
frame
timestamp

Example:

event_onset:
00:13.0

confirmation:
00:20.0

This means:

"The physical incident began at approximately 13 seconds, while sufficient evidence to confidently confirm it accumulated by approximately 20 seconds."

============================================================
28. BUILD THE FINAL CAUSAL CHAIN
============================================================

For the collision example, the final chain should look like:

V_01 Car
   ↓
Approaching V_02
   ↓
Distance decreasing
   ↓
Closing speed increasing
   ↓
CONTACT
00:13.0
   ↓
Relative velocity changes
   ↓
V_02 trajectory deviates
   ↓
V_02 falls
   ↓
Collision confirmed
00:20.0

But this MUST be dynamically generated.

Do not hardcode this chain.

============================================================
29. FINAL CAUSAL GRAPH
============================================================

The final graph should combine:

A. Entity nodes
B. State/event nodes
C. Kinematic evidence
D. Statistical causal edges
E. Temporal event edges

Example:

[Car V_01]
      |
      | approaching
      ↓
[Distance decreasing]
      |
      ↓
[Closing speed increasing]
      |
      ↓
[CONTACT]
  00:13.0
      |
      ↓
[Relative velocity change]
      |
      ↓
[Trajectory deviation]
      |
      ↓
[Bike V_02 falls]
      |
      ↓
[COLLISION CONFIRMED]
  00:20.0

============================================================
30. GRAPH NODE STRUCTURE
============================================================

Each node:

{
    "id": "...",
    "label": "...",
    "type": "object/event/state/evidence",
    "timestamp": 13.0,
    "frame": 393,
    "object_ids": ["V_01", "V_02"],
    "confidence": 0.93
}

============================================================
31. GRAPH EDGE STRUCTURE
============================================================

Each edge:

{
    "source": "...",
    "target": "...",
    "relationship": "causes",
    "confidence": 0.89,
    "lag_seconds": 0.2,
    "evidence": [
        "PCMCI+",
        "LiNGAM",
        "closing_speed",
        "contact"
    ]
}

Use:

causes

only when sufficiently supported.

Otherwise:

precedes

or:

supports.

============================================================
32. EVENT CAUSAL GRAPH JSON
============================================================

Generate a final JSON similar to:

{
  "event_id": "EVT_001",

  "event_type": "COLLISION",

  "event_onset": {
      "frame": 393,
      "timestamp": 13.1
  },

  "confirmation": {
      "frame": 600,
      "timestamp": 20.0
  },

  "entities": [
      {
          "object_id": "V_01",
          "class": "car"
      },
      {
          "object_id": "V_02",
          "class": "motorcycle"
      }
  ],

  "nodes": [],

  "edges": [],

  "statistical_causal_edges": [],

  "temporal_event_edges": [],

  "method_results": {
      "pcmci+": [],
      "lingam": [],
      "ges": [],
      "cyclenet": [],
      "causal_forest": []
  },

  "consensus": [],

  "evidence": [],

  "confidence": 0.93,

  "explanation": ""
}

============================================================
33. FRONTEND GRAPH
============================================================

The project specification requires an interactive D3 causal graph.

Implement:

CausalGraphVisualization

The graph must display:

- objects
- events
- states
- causal edges
- temporal order
- confidence

Hover:

show:

method support
confidence
lag
p-value
timestamp

Click:

show detailed evidence.

If possible:

click event node
→ seek video player to that frame.

============================================================
34. GRAPH SHOULD BE HUMAN-READABLE
============================================================

Do NOT display only:

V_01_Velocity
V_02_Position
V_03_Acceleration

Instead display semantic labels such as:

Vehicle V_01 Speed
Vehicle V_02 Deceleration
Distance V_01–V_02
Closing Speed
Physical Contact
Trajectory Deviation
Collision

Allow technical variable names in a details panel.

============================================================
35. CAUSAL EXPLANATION
============================================================

Generate an explanation directly from the causal graph.

Example:

"Vehicle V_01 was approaching Vehicle V_02 before the incident. The distance between the vehicles decreased and closing speed increased. At approximately 00:13.1, contact was detected together with a sharp relative-velocity change. Vehicle V_02 subsequently deviated from its trajectory and fell. The system accumulated additional evidence until approximately 00:20.0, when the collision was confirmed."

Do not invent facts.

Every sentence must be traceable to graph evidence.

============================================================
36. MOG2 ROLE
============================================================

MOG2 is supporting evidence.

Do NOT use:

MOG2 motion spike
→
collision.

Instead:

MOG2 motion
+
object interaction
+
kinematic change
+
temporal evidence

can strengthen an event hypothesis.

============================================================
37. YOLO ROLE
============================================================

YOLO tells us:

"What objects are visible?"

BoT-SORT tells us:

"Which physical object is which?"

Homography tells us:

"Where is the object in world coordinates?"

Kinematics tells us:

"How is it moving?"

Causal engine tells us:

"How are those changes related?"

Event graph tells us:

"What happened and why?"

============================================================
38. DATA QUALITY
============================================================

The causal engine must track:

observed
interpolated
predicted

values.

Do not treat interpolated data as equivalent to observed data.

A causal relationship supported only by low-confidence interpolated values should receive lower confidence.

============================================================
39. TEMPORAL ALIGNMENT
============================================================

Use original video timing.

Preserve:

original_fps
frame_id
timestamp

Ensure:

timestamp = frame_id / original_fps

or the correct timestamp mechanism for variable timing.

Do not accidentally calculate timestamps from:

processing FPS
CSV row number
batch number
inference completion time.

This is a possible reason why event timestamps can become incorrect.

============================================================
40. API
============================================================

Implement:

POST /api/causal/analyze/{event_id}

GET /api/causal/graph/{event_id}

GET /api/causal/methods

The analyze endpoint should:

1. load kinematic CSV
2. validate data
3. construct entity time series
4. construct pairwise interactions
5. run statistical causal discovery
6. extract temporal events
7. build event-level causal relationships
8. localize event onset
9. construct final causal graph
10. generate explanation
11. save causal_graph.json
12. return result

============================================================
41. SAVE RESULTS
============================================================

Save:

causal_graph.json

method_results.json

consensus_edges.csv

event_timeline.json

causal_report.json

Use the existing dataset/event directory structure.

============================================================
42. METHOD FAILURE HANDLING
============================================================

If one algorithm fails:

DO NOT make the entire engine fail.

Example:

PCMCI+ = successful
LiNGAM = successful
GES = successful
CycleNet = unavailable
Causal Forest = skipped

The engine should still produce a graph.

Clearly report:

available_methods
failed_methods
skipped_methods
failure_reason

============================================================
43. AVOID FALSE CAUSALITY
============================================================

Test cases:

1. Two cars approach but never collide.
2. A bike passes a car.
3. A vehicle brakes normally.
4. Two vehicles independently brake because of a traffic signal.
5. Temporary occlusion.
6. Tracking ID switch.
7. MOG2 motion spike without collision.

The engine must not conclude collision causality from proximity alone.

============================================================
44. REQUIRED TEST: CURRENT PROBLEMATIC VIDEO
============================================================

Run the completed causal engine on the problematic video where:

actual accident:
~00:13

current detected event:
~00:20

Inspect the complete temporal sequence.

The output must contain:

EVENT ONSET:
~00:13.x

EVENT CONFIRMATION:
~00:20.x

Do not use a fixed offset.

Explain exactly which frame/feature caused the onset to be selected.

============================================================
45. FINAL ARCHITECTURE
============================================================

The completed system should be:

VIDEO
 ↓
YOLO
 ↓
BoT-SORT
 ↓
Unique Object IDs
 ↓
Homography
 ↓
World Coordinates
 ↓
Velocity / Acceleration
 ↓
Entity Time Series
 ↓
Pairwise Interaction Features
 ↓
 ┌──────────────────────────┐
 │ CAUSAL ENGINE             │
 │                           │
 │ PCMCI+                    │
 │ LiNGAM                    │
 │ GES                       │
 │ CycleNet (if available)   │
 │ Causal Forest             │
 └────────────┬─────────────┘
              ↓
      Consensus Causal Graph
              ↓
       Temporal Event Engine
              ↓
        Event Causal Graph
              ↓
       Event Onset Localizer
              ↓
      ONSET + CONFIRMATION
              ↓
      Human Explanation
              ↓
      Interactive D3 Graph

============================================================
46. IMPORTANT SCIENTIFIC REQUIREMENT
============================================================

Do NOT claim:

"PCMCI proves the accident was caused by Vehicle A."

Instead report:

"PCMCI+ provides statistical temporal evidence supporting the relationship."

Similarly:

LiNGAM provides directional causal evidence under its assumptions.

GES provides score-based structural evidence.

Physical interaction features provide domain evidence.

Temporal event reasoning provides event-level evidence.

The final consensus should clearly show all sources.

============================================================
47. FINAL IMPLEMENTATION REPORT
============================================================

After implementation, provide:

1. Files created
2. Files modified
3. Exact causal architecture
4. Input data schema
5. Kinematic variables
6. Pairwise variables
7. PCMCI+ implementation
8. LiNGAM implementation
9. GES implementation
10. CycleNet status
11. Causal Forest implementation
12. Consensus mechanism
13. Event extraction mechanism
14. Event causality mechanism
15. Event onset localization mechanism
16. Final graph schema
17. API endpoints
18. Frontend changes
19. Test results
20. Result for the ~00:13 vs ~00:20 problem
21. Known limitations

DO NOT SAY "implemented" unless you actually implemented and tested it.

DO NOT FAKE CAUSAL EDGES.

DO NOT FAKE ALGORITHM RESULTS.

DO NOT HARD-CODE THE ACCIDENT TIMESTAMP.

DO NOT HARD-CODE CAR-BIKE COLLISION.

DO NOT USE ONLY CORRELATION AND CALL IT CAUSALITY.

The final system must genuinely implement:

KINEMATIC CAUSAL DISCOVERY
+
TEMPORAL EVENT CAUSALITY
+
MULTI-METHOD CONSENSUS
+
DYNAMIC CAUSAL GRAPH
+
EVENT ONSET LOCALIZATION.