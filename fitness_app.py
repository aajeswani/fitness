import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

DISCLAIMER = (
    "Disclaimer: This plan does not diagnose, screen for, or rule out any medical condition. "
    "It does not guarantee the prevention of any disease. "
    "Please consult a qualified healthcare provider before making any health or dietary changes."
)


class Goal(Enum):
    WEIGHT_LOSS = "Weight Loss"
    PREVENT_DIABETES = "Prevent Diabetes"


class Sex(Enum):
    MALE = "Male"
    FEMALE = "Female"


class Activity(Enum):
    SEDENTARY = 1.2
    LIGHT = 1.375
    MODERATE = 1.55
    ACTIVE = 1.725
    VERY_ACTIVE = 1.9


class BmiCategory(Enum):
    UNDERWEIGHT = "Underweight"
    HEALTHY = "Healthy"
    OVERWEIGHT = "Overweight"
    OBESE = "Obese"


class RiskBand(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


@dataclass(frozen=True)
class Profile:
    age: float
    weight_kg: float
    height_cm: float
    sex: Sex
    activity: Activity
    family_history: bool
    high_bp: bool
    goal: Goal

    def __post_init__(self) -> None:
        if not math.isfinite(self.age) or not (13.0 <= self.age <= 100.0):
            raise ValueError("Age must be finite and between 13 and 100.")
        if not math.isfinite(self.weight_kg) or not (25.0 <= self.weight_kg <= 400.0):
            raise ValueError("Weight must be finite and between 25 and 400 kg.")
        if not math.isfinite(self.height_cm) or not (100.0 <= self.height_cm <= 250.0):
            raise ValueError("Height must be finite and between 100 and 250 cm.")


@dataclass(frozen=True)
class Metrics:
    bmr: float
    tdee: float
    bmi: float
    bmi_category: BmiCategory
    healthy_weight_ceiling: float


@dataclass(frozen=True)
class WeightLossDetail:
    target_kcal: float
    deficit_kcal: float
    protein_g: int
    is_floored: bool
    weekly_loss_kg: float | None = None
    kg_above_ceiling: float | None = None
    weeks_to_ceiling: float | None = None


@dataclass(frozen=True)
class DiabetesRisk:
    score: int
    band: RiskBand
    factors: tuple[str, ...]
    seven_percent_kg: float | None = None


@dataclass(frozen=True)
class Plan:
    goal: Goal
    metrics: Metrics
    headline: str
    actions: tuple[str, ...]
    weight_loss: WeightLossDetail | None = None
    risk: DiabetesRisk | None = None


def parse_goal(raw: str) -> Goal | None:
    """Parses a user input string into a recognized Goal."""
    val = raw.strip().lower()
    if val in ("1", "weight loss"):
        return Goal.WEIGHT_LOSS
    if val in ("2", "prevent diabetes"):
        return Goal.PREVENT_DIABETES
    return None


def parse_sex(raw: str) -> Sex | None:
    """Parses a user input string into a recognized Sex."""
    val = raw.strip().lower()
    if val in ("m", "male", "1"):
        return Sex.MALE
    if val in ("f", "female", "2"):
        return Sex.FEMALE
    return None


def parse_activity(raw: str) -> Activity | None:
    """Parses a user input string into a recognized Activity."""
    val = raw.strip().lower()
    if val in ("sedentary", "1"):
        return Activity.SEDENTARY
    if val in ("light", "2"):
        return Activity.LIGHT
    if val in ("moderate", "3"):
        return Activity.MODERATE
    if val in ("active", "4"):
        return Activity.ACTIVE
    if val in ("very active", "5"):
        return Activity.VERY_ACTIVE
    return None


def parse_yes_no(raw: str) -> bool | None:
    """Parses a user input string into a boolean."""
    val = raw.strip().lower()
    if val in ("y", "yes", "true", "1"):
        return True
    if val in ("n", "no", "false", "0"):
        return False
    return None


def compute_metrics(profile: Profile) -> Metrics:
    """Computes pure physiological metrics based on the given profile."""
    # BMI calculation and rounding
    bmi_raw = profile.weight_kg / ((profile.height_cm / 100.0) ** 2)
    bmi = round(bmi_raw, 1)

    if bmi < 18.5:
        bmi_cat = BmiCategory.UNDERWEIGHT
    elif bmi < 25.0:
        bmi_cat = BmiCategory.HEALTHY
    elif bmi < 30.0:
        bmi_cat = BmiCategory.OVERWEIGHT
    else:
        bmi_cat = BmiCategory.OBESE

    # Health metrics
    healthy_weight_ceiling = 24.9 * ((profile.height_cm / 100.0) ** 2)

    # BMR via Mifflin-St Jeor
    bmr = (10.0 * profile.weight_kg) + (6.25 * profile.height_cm) - (5.0 * profile.age)
    bmr += 5.0 if profile.sex == Sex.MALE else -161.0
    
    tdee = bmr * profile.activity.value

    return Metrics(
        bmr=bmr,
        tdee=tdee,
        bmi=bmi,
        bmi_category=bmi_cat,
        healthy_weight_ceiling=healthy_weight_ceiling,
    )


def plan_for(profile: Profile) -> Plan:
    """Generates a structured, safety-bounded plan tailored to the stated goal."""
    metrics = compute_metrics(profile)
    protein_g = round(1.6 * profile.weight_kg)

    if profile.goal == Goal.WEIGHT_LOSS:
        floor = 1500.0 if profile.sex == Sex.MALE else 1200.0
        is_floored = metrics.tdee < floor

        if is_floored:
            target_kcal = float(floor)
            deficit_kcal = 0.0
        else:
            target_kcal = min(metrics.tdee, max(metrics.tdee - 500.0, float(floor)))
            deficit_kcal = max(0.0, metrics.tdee - target_kcal)

        weekly_loss_kg = (deficit_kcal * 7.0 / 7700.0) if deficit_kcal > 0.0 else None

        kg_above_ceiling = None
        if profile.weight_kg > metrics.healthy_weight_ceiling:
            kg_above_ceiling = profile.weight_kg - metrics.healthy_weight_ceiling

        weeks_to_ceiling = None
        if weekly_loss_kg is not None and kg_above_ceiling is not None:
            weeks_to_ceiling = kg_above_ceiling / weekly_loss_kg

        weight_loss_detail = WeightLossDetail(
            target_kcal=target_kcal,
            deficit_kcal=deficit_kcal,
            protein_g=protein_g,
            is_floored=is_floored,
            weekly_loss_kg=weekly_loss_kg,
            kg_above_ceiling=kg_above_ceiling,
            weeks_to_ceiling=weeks_to_ceiling,
        )

        actions = ["Maintain a sustainable and safe dietary pattern.", f"Aim for {protein_g}g of protein daily."]
        if is_floored:
            actions.append(
                "Your daily burn is below the safe calorie floor. Consult a clinician or dietitian to set a safe, personalized deficit."
            )

        return Plan(
            goal=profile.goal,
            metrics=metrics,
            headline="Weight Loss Plan",
            actions=tuple(actions),
            weight_loss=weight_loss_detail,
            risk=None,
        )

    else:
        score = 0
        factors = []

        if profile.age >= 60:
            score += 3
            factors.append("Age 60+")
        elif profile.age >= 45:
            score += 2
            factors.append("Age 45-59")

        if metrics.bmi >= 30.0:
            score += 3
            factors.append("BMI 30+ (Obese)")
        elif metrics.bmi >= 25.0:
            score += 2
            factors.append("BMI 25-29.9 (Overweight)")

        if profile.family_history:
            score += 2
            factors.append("Family history of diabetes")

        if profile.high_bp:
            score += 2
            factors.append("High blood pressure")

        if profile.activity == Activity.SEDENTARY:
            score += 2
            factors.append("Sedentary lifestyle")

        if score >= 6:
            band = RiskBand.HIGH
        elif score >= 3:
            band = RiskBand.MODERATE
        else:
            band = RiskBand.LOW

        seven_pct = profile.weight_kg * 0.07 if metrics.bmi >= 25.0 else None

        risk_detail = DiabetesRisk(
            score=score,
            band=band,
            factors=tuple(factors),
            seven_percent_kg=seven_pct,
        )

        actions = ["Maintain regular physical activity and a balanced diet."]
        if band == RiskBand.HIGH:
            actions.append("Consider asking a clinician for an A1C test based on your risk factors.")

        return Plan(
            goal=profile.goal,
            metrics=metrics,
            headline="Diabetes Prevention Plan",
            actions=tuple(actions),
            weight_loss=None,
            risk=risk_detail,
        )


def render(plan: Plan) -> str:
    """Renders the Plan into non-empty user-facing text, gracefully handling optional fields."""
    lines = []
    lines.append(f"--- {plan.headline} ---")
    lines.append(f"BMI: {plan.metrics.bmi} ({plan.metrics.bmi_category.value})")
    lines.append(f"Estimated Daily Burn (TDEE): {plan.metrics.tdee:.0f} kcal")

    if plan.goal == Goal.WEIGHT_LOSS and plan.weight_loss:
        wl = plan.weight_loss
        lines.append(f"Target Intake: {wl.target_kcal:.0f} kcal/day")
        lines.append(f"Daily Deficit: {wl.deficit_kcal:.0f} kcal/day")
        lines.append(f"Protein Target: {wl.protein_g} g/day")

        if wl.weekly_loss_kg is not None:
            lines.append(f"Projected Weekly Loss: {wl.weekly_loss_kg:.2f} kg")
        if wl.kg_above_ceiling is not None:
            lines.append(f"Distance to Healthy Weight: {wl.kg_above_ceiling:.1f} kg")
        if wl.weeks_to_ceiling is not None:
            lines.append(f"Weeks to Healthy Weight: {wl.weeks_to_ceiling:.1f}")

        if wl.is_floored:
            lines.append("Note: No safe self-directed deficit exists at this metabolic rate.")

    elif plan.goal == Goal.PREVENT_DIABETES and plan.risk:
        rk = plan.risk
        lines.append(f"Risk Score: {rk.score} ({rk.band.value} Risk)")
        if rk.factors:
            lines.append("Identified Factors: " + ", ".join(rk.factors))
        
        if rk.seven_percent_kg is not None:
            lines.append(f"Target 7% Weight Loss: {rk.seven_percent_kg:.1f} kg")

    if plan.actions:
        lines.append("Recommended Actions:")
        for action in plan.actions:
            lines.append(f" - {action}")

    lines.append("")
    lines.append(DISCLAIMER)

    return "\n".join(lines)


def run_intake(ask: Callable, *, show: Callable | None = None) -> Profile:
    """Interactively collects a valid Profile from the user."""
    if show is None:
        show = print

    while True:
        goal = None
        while goal is None:
            goal = parse_goal(ask("Select goal (1 for Weight Loss, 2 for Prevent Diabetes): "))

        def ask_float(prompt_str: str) -> float | None:
            try:
                return float(ask(prompt_str))
            except ValueError:
                return None

        age = None
        while age is None:
            age = ask_float("Age (13-100): ")

        weight_kg = None
        while weight_kg is None:
            weight_kg = ask_float("Weight in kg (25-400): ")

        height_cm = None
        while height_cm is None:
            height_cm = ask_float("Height in cm (100-250): ")

        sex = None
        while sex is None:
            sex = parse_sex(ask("Sex (Male/Female): "))

        activity = None
        while activity is None:
            activity = parse_activity(ask("Activity level (Sedentary/Light/Moderate/Active/Very Active): "))

        family_history = None
        while family_history is None:
            family_history = parse_yes_no(ask("Family history of diabetes? (y/n): "))

        high_bp = None
        while high_bp is None:
            high_bp = parse_yes_no(ask("High blood pressure? (y/n): "))

        try:
            return Profile(
                age=age,
                weight_kg=weight_kg,
                height_cm=height_cm,
                sex=sex,
                activity=activity,
                family_history=family_history,
                high_bp=high_bp,
                goal=goal,
            )
        except ValueError as e:
            show(f"Validation error: {e}")
            show("Please restart and provide valid measurements.\n")


def main(ask: Callable = input, show: Callable = print) -> int:
    """CLI entry point. Runs intake, generates a plan, and renders it."""
    try:
        profile = run_intake(ask, show=show)
        plan = plan_for(profile)
        output = render(plan)
        show(output)
        return 0
    except (KeyboardInterrupt, EOFError):
        return 1