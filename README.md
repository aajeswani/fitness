# Product: Fitfood

## Assumptions

This project supports exactly two user goals: **Weight Loss**, **Muscle Gain**, **Gain Mass** and **Prevent Diabetes**. Users provide a short profile including age, sex, height, weight, activity level, fitness goal, and medical history (e.g., family history, blood-pressure and chronic history, if any) Each request is processed as a single stateless evaluation. The application does not store, transmit, or log health information, and it does not diagnose, screen for, treat, or guarantee prevention of any medical condition.

## Objective

The objective is to generate a structured and safety-bounded plan based on the user’s selected goal and profile. For weight loss, the module calculates BMI, BMR, TDEE, a calorie target, expected weekly weight change, and a protein target while enforcing minimum calorie floors. For diabetes prevention, it produces a non-diagnostic risk-awareness score, identifies contributing factors, and may suggest discussing an A1C test with a healthcare professional.

## Design

The project separates user intake from safety-bearing calculation logic. `run_intake()` collects and validates responses, while `plan_for()` performs the core planning calculations as a pure function with no input/output, logging, clock access, persistence, or argument mutation.

Profiles, plans, metrics, and goal-specific results use frozen dataclasses so they cannot be modified after creation. Enums define closed sets such as goal, sex, activity level, BMI category, and risk band. Invalid values are rejected instead of guessed or silently converted.

The source prompt defines the module’s responsibility, non-responsibilities, vocabulary, interfaces, safety rules, and acceptance criteria. When behavior must change, the source prompt should be updated and the module regenerated instead of manually patching generated code.

## Current Scope

The current module creates weight-loss and diabetes-risk-awareness plans. It does not yet retrieve nearby restaurants, access live menus, recommend specific foods, use the current time, track previous meals, or store progress over time.

These features should be implemented as separate testable modules before being integrated into the full meal-recommendation application.

Potential future prompt modules include:

- `fitfood_python.prompt`

## Prompt File

The source prompt used to define the module is:

`fitfood_python.prompt`

## Other Deliverables
Presentation Deck: https://docs.google.com/presentation/d/13_ZHqNGmpSwruQ_0HdOmqKmW-rsISsMd/edit?usp=drivesdk&ouid=114297852747829232460&rtpof=true&sd=true
Demo Video: https://youtu.be/uE8RAju1onM?is=3TRIY5wt8_MWuRG-

## How This Project Addresses the Five Judging Criteria

### 1. PDD Method and Traceability

The prompt is treated as the source of truth rather than a one-time coding instruction. It contains explicit numbered contract rules, including required behavior, prohibited behavior, calculation formulas, input boundaries, privacy requirements, and medical-safety limitations.

The intended traceability flow is:

`Source Prompt → Generated Code → Acceptance Tests → Observed Result → Prompt Update → Regeneration`

Examples of traceable requirements include:

- R1 defines the only accepted goals.
- R3 defines how BMI must be rounded and categorized.
- R4 and R5 define calorie-floor and deficit behavior.
- R8 defines diabetes-risk scoring.
- R9 prohibits diagnosis and requires a disclaimer.
- R10 defines input-validation boundaries.
- R11 requires the core planning logic to remain pure.

Because each behavior is tied to a named rule, failures can be traced back to the source prompt and corrected at the specification level.

### 2. Technical Execution

The implementation is designed for reliability, reproducibility, and testing. Safety-bearing calculations are isolated in pure functions, user input is handled through injected callables, and rendering returns a string rather than directly printing output.

Technical design choices include:

- frozen dataclasses for immutable profiles and plans;
- explicit validation of plausible human measurements;
- rejection of `NaN`, infinity, and invalid numeric inputs;
- deterministic BMI, BMR, TDEE, calorie, protein, and risk calculations;
- no hidden clock, storage, network, or logging dependencies;
- parsers that return `None` rather than guessing invalid answers;
- a structured `Plan` that can contain only the detail type associated with the selected goal.

These choices allow the core behavior to be tested without a terminal or external environment.

### 3. Problem Fit and User Value

The project addresses a specific user need: turning a short health profile and a selected goal into a clear, structured next-step plan.

Instead of providing generic health advice, the module calculates results based on the user’s own profile while applying explicit safety boundaries. It also limits its scope deliberately by supporting only two goals and refusing to diagnose medical conditions.

The value of the current module is that it converts complex health-related calculations into an understandable and reproducible result while clearly explaining its limitations.

### 4. Innovation and Learning

The innovation is not simply generating code with AI. The project uses Prompt Driven Development to turn the prompt into a durable and testable software specification.

The source prompt combines:

- functional requirements;
- safety rules;
- privacy boundaries;
- medical-claim restrictions;
- calculation formulas;
- interface definitions;
- data-model constraints;
- acceptance criteria.

This allows the team to improve behavior through evidence-led prompt iteration. When a generated result fails a requirement, the team can identify the relevant rule, revise the source prompt, regenerate the module, and rerun the tests while keeping intent, code, and validation synchronized.

### 5. Demo and Communication

The module produces structured user-facing output that explains the user’s goal, calculated metrics, recommended actions, and safety disclaimer.

The demo can clearly show:

1. the user selecting a supported goal;
2. the application collecting and validating a profile;
3. the planning module calculating a result;
4. the rendered plan explaining the output;
5. invalid or unsafe inputs being rejected;
6. the prompt rule connected to the demonstrated behavior.

The project also communicates its scope honestly. It distinguishes completed functionality from future product capabilities and does not claim clinical validation or guaranteed health outcomes.

### 6. Slides

https://docs.google.com/presentation/d/13_ZHqNGmpSwruQ_0HdOmqKmW-rsISsMd/edit?slide=id.p1#slide=id.p1

### 7. Demo snapshot

https://github.com/aajeswani/fitness/blob/main/demo.png

## Privacy and Safety

The module is stateless and does not persist, transmit, or log user health information. It enforces calorie floors, rejects implausible inputs, avoids unsupervised medical claims, and includes a disclaimer in every rendered plan.

## Disclaimer

This project is for educational and general planning purposes only. It is not medical advice and does not diagnose, treat, screen for, or guarantee prevention of any condition. Users should consult a qualified healthcare professional before making significant health or dietary changes.
