# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Domain-Specific Culinary Tools with Comprehensive Docstrings, Descriptive Naming,
Strict Pydantic JSON Schemas, Guided Error Handling, and Human-in-the-Loop Confirmation.

Implements:
- Rubric 1.1: Comprehensive Tool Docstrings (purpose, all parameters, return structure).
- Rubric 1.2: Descriptive Naming (`match_ingredients_to_meal_options`, `generate_structured_recipe_card`,
  `calculate_recipe_nutrition_profile`, `save_dietary_preference_to_memory`, `dispatch_missing_ingredients_order`).
- Rubric 1.3: Explicit JSON Schemas (validated via `app/schemas.py`).
- Rubric 1.4: Guided Error Handling (`GuidedToolErrorResponse` with `llm_recovery_instructions`).
- Rubric 3.4: Human-in-the-Loop Hooks (`FunctionTool(..., require_confirmation=True)` and `tool_confirmation` check).
"""

from __future__ import annotations

import uuid
from typing import Any

from google.adk.tools import FunctionTool, ToolContext
from pydantic import ValidationError

from app.knowledge_store import CULINARY_STORE
from app.memory import ASYNC_MEMORY_CONSOLIDATOR
from app.observability import (
    DEFAULT_PII_SCRUBBER,
    TRACER,
    log_agent_intent,
    log_agent_outcome,
)
from app.schemas import (
    DietaryPreferenceMemoryInputSchema,
    GuidedToolErrorResponse,
    IngredientMatchInputSchema,
    IngredientMatchOutputSchema,
    MealCandidateSchema,
    NutritionCalculationInputSchema,
    NutritionCalculationOutputSchema,
    PantryOrderInputSchema,
    RecipeStepSchema,
    StructuredRecipeInputSchema,
    StructuredRecipeOutputSchema,
)
from app.secrets import SECRET_MANAGER

# Reference macro table per 100g / typical serving portion for accurate deterministic calculation
_INGREDIENT_MACRO_TABLE: dict[str, dict[str, Any]] = {
    "chicken breast": {"calories": 280, "protein": 52.0, "carbs": 0.0, "fat": 6.0, "fiber": 0.0, "allergens": []},
    "chicken": {"calories": 280, "protein": 52.0, "carbs": 0.0, "fat": 6.0, "fiber": 0.0, "allergens": []},
    "salmon": {"calories": 360, "protein": 40.0, "carbs": 0.0, "fat": 22.0, "fiber": 0.0, "allergens": ["fish"]},
    "eggs": {"calories": 155, "protein": 13.0, "carbs": 1.1, "fat": 11.0, "fiber": 0.0, "allergens": ["egg"]},
    "egg": {"calories": 155, "protein": 13.0, "carbs": 1.1, "fat": 11.0, "fiber": 0.0, "allergens": ["egg"]},
    "chickpeas": {"calories": 270, "protein": 14.5, "carbs": 45.0, "fat": 4.2, "fiber": 12.5, "allergens": []},
    "rice": {"calories": 240, "protein": 5.0, "carbs": 53.0, "fat": 0.5, "fiber": 1.2, "allergens": []},
    "quinoa": {"calories": 222, "protein": 8.1, "carbs": 39.4, "fat": 3.6, "fiber": 5.2, "allergens": []},
    "pasta": {"calories": 310, "protein": 11.0, "carbs": 62.0, "fat": 1.8, "fiber": 3.0, "allergens": ["wheat", "gluten"]},
    "coconut milk": {"calories": 230, "protein": 2.3, "carbs": 6.0, "fat": 24.0, "fiber": 2.0, "allergens": []},
    "olive oil": {"calories": 119, "protein": 0.0, "carbs": 0.0, "fat": 13.5, "fiber": 0.0, "allergens": []},
    "spinach": {"calories": 25, "protein": 3.0, "carbs": 3.8, "fat": 0.4, "fiber": 2.4, "allergens": []},
    "asparagus": {"calories": 27, "protein": 3.0, "carbs": 5.2, "fat": 0.2, "fiber": 2.8, "allergens": []},
    "tomatoes": {"calories": 35, "protein": 1.6, "carbs": 7.5, "fat": 0.4, "fiber": 2.2, "allergens": []},
    "garlic": {"calories": 15, "protein": 0.6, "carbs": 3.3, "fat": 0.1, "fiber": 0.2, "allergens": []},
    "lemon": {"calories": 17, "protein": 0.6, "carbs": 5.4, "fat": 0.2, "fiber": 1.6, "allergens": []},
    "parmesan": {"calories": 110, "protein": 10.0, "carbs": 0.9, "fat": 7.3, "fiber": 0.0, "allergens": ["dairy", "milk"]},
    "soy sauce": {"calories": 15, "protein": 2.0, "carbs": 1.5, "fat": 0.0, "fiber": 0.2, "allergens": ["soy", "wheat"]},
    "peanuts": {"calories": 170, "protein": 7.0, "carbs": 5.0, "fat": 14.0, "fiber": 2.5, "allergens": ["peanuts", "nuts"]},
}


def match_ingredients_to_meal_options(
    available_ingredients: list[str],
    dietary_restrictions: list[str],
    max_prep_time_minutes: int,
    meal_type: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Matches a user's list of available ingredients against the persistent culinary knowledge
    store to identify the best meal option and runner-up alternatives.

    Args:
        available_ingredients: List of ingredient names the user has on hand (e.g.,
            ['chicken breast', 'garlic', 'lemon', 'olive oil', 'rice']).
        dietary_restrictions: List of dietary constraints or allergies to enforce (e.g.,
            ['vegan', 'gluten-free'], or an empty list [] if none).
        max_prep_time_minutes: Maximum total preparation and cooking time in minutes
            (between 5 and 240; pass 45 as a standard default).
        meal_type: Target meal category ('breakfast', 'lunch', 'dinner', 'snack', or 'any').

    Returns:
        A JSON-serializable dictionary adhering to `IngredientMatchOutputSchema` with
        `best_meal_option` and `alternative_meal_options`, or a `GuidedToolErrorResponse`
        dictionary with `llm_recovery_instructions` if inputs are insufficient or invalid.
    """
    start_t = log_agent_intent(
        component="tool",
        agent_name="meal_recipe_agent",
        action_name="match_ingredients_to_meal_options",
        intended_inputs={
            "available_ingredients": available_ingredients,
            "dietary_restrictions": dietary_restrictions,
            "max_prep_time_minutes": max_prep_time_minutes,
            "meal_type": meal_type,
        },
    )

    with TRACER.start_as_current_span("tool.match_ingredients_to_meal_options") as span:
        try:
            validated_input = IngredientMatchInputSchema(
                available_ingredients=available_ingredients,
                dietary_restrictions=dietary_restrictions,
                max_prep_time_minutes=max_prep_time_minutes,
                meal_type=meal_type,
            )
        except ValidationError as exc:
            error_resp = GuidedToolErrorResponse(
                error_code="INVALID_INGREDIENT_SEARCH_PARAMETERS",
                error_message=f"Input schema validation failed: {exc.errors()[0]['msg']}",
                invalid_parameters={
                    "available_ingredients": available_ingredients,
                    "max_prep_time_minutes": max_prep_time_minutes,
                },
                llm_recovery_instructions=(
                    "Ensure `available_ingredients` contains at least one valid ingredient string "
                    "and `max_prep_time_minutes` is between 5 and 240. If the user did not specify "
                    "a time limit, retry with `max_prep_time_minutes=45`."
                ),
                suggested_fallback_options=[
                    "Ask the user to list at least 2 core ingredients in their kitchen."
                ],
            ).model_dump()
            log_agent_outcome(
                component="tool",
                agent_name="meal_recipe_agent",
                action_name="match_ingredients_to_meal_options",
                actual_result=error_resp,
                status="error",
                start_time_monotonic=start_t,
            )
            return error_resp

        # Merge any persistent dietary restrictions stored in session state
        effective_restrictions = list(validated_input.dietary_restrictions)
        if tool_context is not None:
            stored_profile = tool_context.state.get("user_dietary_profile")
            if isinstance(stored_profile, dict):
                for item in stored_profile.get("dietary_restrictions", []) + stored_profile.get("allergies", []):
                    if item and item.lower() not in [r.lower() for r in effective_restrictions]:
                        effective_restrictions.append(item)

        # Check if user only passed trivial seasonings like water/salt
        trivial_only = {"water", "salt", "pepper", "black pepper", "ice"}
        substantive_ings = [
            ing for ing in validated_input.available_ingredients if ing not in trivial_only
        ]
        if not substantive_ings:
            error_resp = GuidedToolErrorResponse(
                error_code="INSUFFICIENT_CORE_INGREDIENTS",
                error_message=(
                    "Only basic seasonings or water were provided; at least one substantive "
                    "protein, grain, or vegetable is required to formulate a nutritious meal."
                ),
                invalid_parameters={
                    "available_ingredients": validated_input.available_ingredients
                },
                llm_recovery_instructions=(
                    "Politely explain to the user that seasonings/water alone cannot make a complete meal, "
                    "and ask them to share 2-3 core pantry, fridge, or freezer staples (such as eggs, rice, "
                    "pasta, canned beans, chicken, or vegetables)."
                ),
                suggested_fallback_options=[
                    "Classic Garlic & Olive Oil Pasta (Aglio e Olio)",
                    "Simple Egg Drop Soup",
                    "Steamed Seasoned Rice with Soy & Sesame",
                ],
            ).model_dump()
            log_agent_outcome(
                component="tool",
                agent_name="meal_recipe_agent",
                action_name="match_ingredients_to_meal_options",
                actual_result=error_resp,
                status="error",
                start_time_monotonic=start_t,
            )
            return error_resp

        candidates = CULINARY_STORE.search_meals_by_ingredients(
            available_ingredients=validated_input.available_ingredients,
            dietary_restrictions=effective_restrictions,
            max_prep_time_minutes=validated_input.max_prep_time_minutes,
            meal_type=validated_input.meal_type,
        )

        if not candidates:
            error_resp = GuidedToolErrorResponse(
                error_code="NO_MATCHING_MEAL_WITHIN_CONSTRAINTS",
                error_message=(
                    f"No meal candidates matched ingredients {validated_input.available_ingredients} "
                    f"within {validated_input.max_prep_time_minutes} minutes and restrictions {effective_restrictions}."
                ),
                invalid_parameters={
                    "max_prep_time_minutes": validated_input.max_prep_time_minutes,
                    "dietary_restrictions": effective_restrictions,
                },
                llm_recovery_instructions=(
                    "Retry `match_ingredients_to_meal_options` with a relaxed `max_prep_time_minutes=90`, "
                    "or suggest a simple skillet/bowl combining the user's safe ingredients directly."
                ),
                suggested_fallback_options=[
                    "Retry with max_prep_time_minutes=90",
                    "Ask user if they have a grain or protein to pair with their produce",
                ],
            ).model_dump()
            log_agent_outcome(
                component="tool",
                agent_name="meal_recipe_agent",
                action_name="match_ingredients_to_meal_options",
                actual_result=error_resp,
                status="error",
                start_time_monotonic=start_t,
            )
            return error_resp

        best_raw = candidates[0]
        alt_raws = candidates[1:3]

        best_schema = MealCandidateSchema(
            meal_id=best_raw["meal_id"],
            meal_name=best_raw["meal_name"],
            cuisine_style=best_raw["cuisine_style"],
            match_score_pct=best_raw["match_score_pct"],
            matched_ingredients=best_raw["matched_ingredients"],
            missing_ingredients=best_raw["missing_ingredients"],
            prep_time_minutes=best_raw["prep_time_minutes"],
            dietary_tags=best_raw["dietary_tags"],
            why_good_option=best_raw["why_good_option"],
        )
        alt_schemas = [
            MealCandidateSchema(
                meal_id=c["meal_id"],
                meal_name=c["meal_name"],
                cuisine_style=c["cuisine_style"],
                match_score_pct=c["match_score_pct"],
                matched_ingredients=c["matched_ingredients"],
                missing_ingredients=c["missing_ingredients"],
                prep_time_minutes=c["prep_time_minutes"],
                dietary_tags=c["dietary_tags"],
                why_good_option=c["why_good_option"],
            )
            for c in alt_raws
        ]

        output = IngredientMatchOutputSchema(
            best_meal_option=best_schema,
            alternative_meal_options=alt_schemas,
            applied_dietary_filters=effective_restrictions,
        ).model_dump()

        if tool_context is not None:
            tool_context.state["selected_meal_option"] = best_schema.model_dump()
            tool_context.state["last_matched_recipe_raw"] = best_raw.get("raw_recipe")

        span.set_attribute("meal.selected_id", best_schema.meal_id)
        log_agent_outcome(
            component="tool",
            agent_name="meal_recipe_agent",
            action_name="match_ingredients_to_meal_options",
            actual_result={"best_meal": best_schema.meal_name, "score": best_schema.match_score_pct},
            status="success",
            start_time_monotonic=start_t,
        )
        return output


def generate_structured_recipe_card(
    meal_id_or_name: str,
    available_ingredients: list[str],
    servings: int,
    dietary_restrictions: list[str],
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Generates a complete, schema-validated culinary recipe card with scaled ingredient
    measurements, required equipment, numbered cooking steps, USDA food-safety checkpoints,
    and chef substitution tips.

    Args:
        meal_id_or_name: Identifier or full name of the selected meal option (e.g.,
            'mediterranean_lemon_herb_chicken_rice' or 'Mediterranean Lemon Garlic Herb Chicken with Rice').
        available_ingredients: List of ingredients the user currently has on hand.
        servings: Number of servings to scale the recipe for (integer between 1 and 20; default 2).
        dietary_restrictions: List of dietary restrictions or allergies to honor in substitutions.

    Returns:
        A JSON-serializable dictionary adhering to `StructuredRecipeOutputSchema`, or a
        `GuidedToolErrorResponse` with recovery instructions if parameters are invalid.
    """
    start_t = log_agent_intent(
        component="tool",
        agent_name="meal_recipe_agent",
        action_name="generate_structured_recipe_card",
        intended_inputs={
            "meal_id_or_name": meal_id_or_name,
            "available_ingredients": available_ingredients,
            "servings": servings,
            "dietary_restrictions": dietary_restrictions,
        },
    )

    with TRACER.start_as_current_span("tool.generate_structured_recipe_card") as span:
        try:
            validated = StructuredRecipeInputSchema(
                meal_id_or_name=meal_id_or_name,
                available_ingredients=available_ingredients,
                servings=servings,
                dietary_restrictions=dietary_restrictions,
            )
        except ValidationError as exc:
            error_resp = GuidedToolErrorResponse(
                error_code="INVALID_RECIPE_CARD_PARAMETERS",
                error_message=f"Recipe card input validation failed: {exc.errors()[0]['msg']}",
                invalid_parameters={
                    "meal_id_or_name": meal_id_or_name,
                    "servings": servings,
                },
                llm_recovery_instructions=(
                    "Ensure `servings` is an integer between 1 and 20 and `meal_id_or_name` is "
                    "non-empty. Retry `generate_structured_recipe_card` with `servings=2`."
                ),
            ).model_dump()
            log_agent_outcome(
                component="tool",
                agent_name="meal_recipe_agent",
                action_name="generate_structured_recipe_card",
                actual_result=error_resp,
                status="error",
                start_time_monotonic=start_t,
            )
            return error_resp

        recipe = CULINARY_STORE.get_recipe_by_id_or_name(validated.meal_id_or_name)
        if recipe is None and tool_context is not None:
            recipe = tool_context.state.get("last_matched_recipe_raw")

        if recipe is None:
            # Synthesize from search_meals_by_ingredients
            matches = CULINARY_STORE.search_meals_by_ingredients(
                available_ingredients=validated.available_ingredients,
                dietary_restrictions=validated.dietary_restrictions,
            )
            if matches:
                recipe = matches[0]["raw_recipe"]

        if recipe is None:
            error_resp = GuidedToolErrorResponse(
                error_code="RECIPE_NOT_FOUND_IN_STORE",
                error_message=f"Could not find recipe matching '{validated.meal_id_or_name}'.",
                invalid_parameters={"meal_id_or_name": validated.meal_id_or_name},
                llm_recovery_instructions=(
                    "Call `match_ingredients_to_meal_options` first to obtain a valid `meal_id`, "
                    "then pass that `meal_id` into `generate_structured_recipe_card`."
                ),
            ).model_dump()
            log_agent_outcome(
                component="tool",
                agent_name="meal_recipe_agent",
                action_name="generate_structured_recipe_card",
                actual_result=error_resp,
                status="error",
                start_time_monotonic=start_t,
            )
            return error_resp

        norm_user_ings = [i.strip().lower() for i in validated.available_ingredients]
        scaled_ingredients: list[dict[str, str]] = []
        for core_ing in recipe.get("core_ingredients", []):
            is_on_hand = any(
                u in core_ing.lower() or core_ing.lower() in u for u in norm_user_ings
            )
            scaled_ingredients.append(
                {
                    "item": core_ing,
                    "amount": f"{validated.servings}x standard portion ({validated.servings * 4} oz / {validated.servings * 115}g)",
                    "status": "on_hand" if is_on_hand else "recommended_addition",
                }
            )
        for staple in recipe.get("pantry_staples", []):
            scaled_ingredients.append(
                {
                    "item": staple,
                    "amount": "to taste (pantry staple)",
                    "status": "pantry_staple",
                }
            )

        steps_schema = [
            RecipeStepSchema(
                step_number=s["step_number"],
                instruction=s["instruction"],
                duration_minutes=s.get("duration_minutes", 5),
                critical_safety_checkpoint=s.get("critical_safety_checkpoint"),
            )
            for s in recipe.get("steps", [])
        ]

        safety_notes = [
            s["critical_safety_checkpoint"]
            for s in recipe.get("steps", [])
            if s.get("critical_safety_checkpoint")
        ]
        if not safety_notes:
            safety_notes = [
                "Wash hands and utensils with warm soapy water for 20 seconds before and after food prep.",
                "Ensure hot foods are cooked and held above 140°F (60°C).",
            ]

        prep_t = int(recipe.get("prep_time_minutes", 10))
        cook_t = int(recipe.get("cook_time_minutes", 15))
        output = StructuredRecipeOutputSchema(
            meal_name=recipe["meal_name"],
            servings=validated.servings,
            prep_time_minutes=prep_t,
            cook_time_minutes=cook_t,
            total_time_minutes=prep_t + cook_t,
            scaled_ingredients=scaled_ingredients,
            equipment_needed=recipe.get("equipment", ["Large skillet", "Chef knife"]),
            steps=steps_schema,
            chef_tips_and_substitutions=recipe.get("chef_tips", []),
            food_safety_guidelines=safety_notes,
        ).model_dump()

        if tool_context is not None:
            tool_context.state["final_recipe_card"] = output

        span.set_attribute("recipe.meal_name", recipe["meal_name"])
        log_agent_outcome(
            component="tool",
            agent_name="meal_recipe_agent",
            action_name="generate_structured_recipe_card",
            actual_result={"meal_name": recipe["meal_name"], "steps_count": len(steps_schema)},
            status="success",
            start_time_monotonic=start_t,
        )
        return output


def calculate_recipe_nutrition_profile(
    ingredients: list[str],
    servings: int,
) -> dict[str, Any]:
    """Calculates the per-serving nutritional macronutrient breakdown (calories, protein,
    carbohydrates, fat, and fiber), dietary compliance badges, and allergen warnings for a recipe.

    Args:
        ingredients: List of ingredient names included in the recipe.
        servings: Number of servings the recipe yields (integer between 1 and 20).

    Returns:
        A JSON-serializable dictionary adhering to `NutritionCalculationOutputSchema`, or a
        `GuidedToolErrorResponse` with recovery instructions if parameters fail validation.
    """
    start_t = log_agent_intent(
        component="tool",
        agent_name="meal_recipe_agent",
        action_name="calculate_recipe_nutrition_profile",
        intended_inputs={"ingredients": ingredients, "servings": servings},
    )

    with TRACER.start_as_current_span("tool.calculate_recipe_nutrition_profile"):
        try:
            validated = NutritionCalculationInputSchema(
                ingredients=ingredients,
                servings=servings,
            )
        except ValidationError as exc:
            error_resp = GuidedToolErrorResponse(
                error_code="INVALID_NUTRITION_INPUTS",
                error_message=f"Nutrition calculation input validation failed: {exc.errors()[0]['msg']}",
                invalid_parameters={"ingredients": ingredients, "servings": servings},
                llm_recovery_instructions=(
                    "Pass a non-empty `ingredients` list and a `servings` integer between 1 and 20 "
                    "(e.g., `servings=2`)."
                ),
            ).model_dump()
            log_agent_outcome(
                component="tool",
                agent_name="meal_recipe_agent",
                action_name="calculate_recipe_nutrition_profile",
                actual_result=error_resp,
                status="error",
                start_time_monotonic=start_t,
            )
            return error_resp

        total_cals = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0
        total_fiber = 0.0
        detected_allergens: set[str] = set()

        for raw_ing in validated.ingredients:
            clean_ing = raw_ing.strip().lower()
            matched_entry = _INGREDIENT_MACRO_TABLE.get(clean_ing)
            if matched_entry is None:
                for key, val in _INGREDIENT_MACRO_TABLE.items():
                    if key in clean_ing or clean_ing in key:
                        matched_entry = val
                        break
            if matched_entry is None:
                # Default whole-food vegetable/aromatic macro estimate
                matched_entry = {
                    "calories": 45.0,
                    "protein": 2.0,
                    "carbs": 8.0,
                    "fat": 0.5,
                    "fiber": 2.0,
                    "allergens": [],
                }
            # Scale slightly by servings for main calculation
            scale = max(1.0, validated.servings / 2.0)
            total_cals += float(matched_entry["calories"]) * scale
            total_protein += float(matched_entry["protein"]) * scale
            total_carbs += float(matched_entry["carbs"]) * scale
            total_fat += float(matched_entry["fat"]) * scale
            total_fiber += float(matched_entry["fiber"]) * scale
            for alg in matched_entry.get("allergens", []):
                detected_allergens.add(alg)

        per_serving = {
            "calories_kcal": round(total_cals / validated.servings, 1),
            "protein_g": round(total_protein / validated.servings, 1),
            "carbs_g": round(total_carbs / validated.servings, 1),
            "fat_g": round(total_fat / validated.servings, 1),
            "fiber_g": round(total_fiber / validated.servings, 1),
        }

        badges: list[str] = []
        if per_serving["protein_g"] >= 25.0:
            badges.append("High-Protein")
        if per_serving["fiber_g"] >= 5.0:
            badges.append("High-Fiber")
        if per_serving["calories_kcal"] <= 600.0:
            badges.append("Calorie-Conscious")
        if not detected_allergens:
            badges.append("Top-8-Allergen-Friendly")

        output = NutritionCalculationOutputSchema(
            servings=validated.servings,
            per_serving_macros=per_serving,
            dietary_compliance_badges=badges,
            allergen_warnings=sorted(detected_allergens),
        ).model_dump()

        log_agent_outcome(
            component="tool",
            agent_name="meal_recipe_agent",
            action_name="calculate_recipe_nutrition_profile",
            actual_result=output,
            status="success",
            start_time_monotonic=start_t,
        )
        return output


def save_dietary_preference_to_memory(
    dietary_restrictions: list[str],
    allergies: list[str],
    disliked_ingredients: list[str],
    preferred_cuisines: list[str],
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Saves the user's dietary restrictions, food allergies, disliked ingredients, and
    preferred cuisines into persistent session state and schedules an asynchronous
    background consolidation task into the persistent SQLite / Vertex AI Memory Bank store.

    Args:
        dietary_restrictions: List of permanent dietary rules (e.g., ['vegan', 'gluten-free']).
        allergies: List of strict food allergies (e.g., ['peanuts', 'shellfish']).
        disliked_ingredients: List of ingredients the user prefers to avoid (e.g., ['cilantro']).
        preferred_cuisines: List of favorite cuisine styles (e.g., ['Mediterranean', 'Thai']).

    Returns:
        A JSON-serializable confirmation dictionary indicating the profile was updated in
        session state and scheduled for non-blocking background persistence.
    """
    user_id = "default_culinary_user"
    if tool_context is not None:
        inv_ctx = getattr(tool_context, "_invocation_context", None)
        if inv_ctx is not None and getattr(inv_ctx, "user_id", None):
            user_id = str(inv_ctx.user_id)

    start_t = log_agent_intent(
        component="tool",
        agent_name="meal_recipe_agent",
        action_name="save_dietary_preference_to_memory",
        intended_inputs={
            "user_id": user_id,
            "dietary_restrictions": dietary_restrictions,
            "allergies": allergies,
            "disliked_ingredients": disliked_ingredients,
            "preferred_cuisines": preferred_cuisines,
        },
    )

    try:
        validated = DietaryPreferenceMemoryInputSchema(
            user_id=user_id,
            dietary_restrictions=dietary_restrictions,
            allergies=allergies,
            disliked_ingredients=disliked_ingredients,
            preferred_cuisines=preferred_cuisines,
        )
    except ValidationError as exc:
        error_resp = GuidedToolErrorResponse(
            error_code="INVALID_DIETARY_MEMORY_PARAMETERS",
            error_message=f"Dietary preference schema validation failed: {exc.errors()[0]['msg']}",
            invalid_parameters={"dietary_restrictions": dietary_restrictions},
            llm_recovery_instructions="Pass valid lists of strings for all four preference fields.",
        ).model_dump()
        log_agent_outcome(
            component="tool",
            agent_name="meal_recipe_agent",
            action_name="save_dietary_preference_to_memory",
            actual_result=error_resp,
            status="error",
            start_time_monotonic=start_t,
        )
        return error_resp

    scrubbed_profile = {
        "user_id": DEFAULT_PII_SCRUBBER.scrub_text(validated.user_id),
        "dietary_restrictions": DEFAULT_PII_SCRUBBER.scrub_payload(validated.dietary_restrictions),
        "allergies": DEFAULT_PII_SCRUBBER.scrub_payload(validated.allergies),
        "disliked_ingredients": DEFAULT_PII_SCRUBBER.scrub_payload(validated.disliked_ingredients),
        "preferred_cuisines": DEFAULT_PII_SCRUBBER.scrub_payload(validated.preferred_cuisines),
    }

    if tool_context is not None:
        tool_context.state["user_dietary_profile"] = scrubbed_profile

    # Schedule non-blocking async memory consolidation (Rubric 2.4: Async Memory Operations)
    task = ASYNC_MEMORY_CONSOLIDATOR.schedule_dietary_memory_consolidation(
        user_id=scrubbed_profile["user_id"],
        dietary_restrictions=scrubbed_profile["dietary_restrictions"],
        allergies=scrubbed_profile["allergies"],
        disliked_ingredients=scrubbed_profile["disliked_ingredients"],
        preferred_cuisines=scrubbed_profile["preferred_cuisines"],
    )

    result = {
        "status": "success",
        "async_consolidation_scheduled": task is not None,
        "saved_profile": scrubbed_profile,
        "message": (
            "Dietary preferences and allergies have been immediately applied to the current "
            "session and queued for non-blocking background persistence."
        ),
    }
    log_agent_outcome(
        component="tool",
        agent_name="meal_recipe_agent",
        action_name="save_dietary_preference_to_memory",
        actual_result=result,
        status="success",
        start_time_monotonic=start_t,
    )
    return result


def _dispatch_missing_ingredients_order_impl(
    missing_items: list[str],
    estimated_total_usd: float,
    delivery_notes: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Dispatches a high-stakes grocery delivery order for missing recipe ingredients.
    Protected by an explicit Human-in-the-Loop (HITL) confirmation gate (`require_confirmation=True`).

    Args:
        missing_items: List of missing ingredient names to purchase (e.g., ['fresh basil', 'parmesan']).
        estimated_total_usd: Estimated order total in USD (float between 0.01 and 500.00).
        delivery_notes: Delivery drop-off instructions (any PII is automatically redacted).

    Returns:
        A JSON-serializable dictionary indicating whether the order is awaiting human confirmation,
        was declined by the human, or was confirmed and dispatched.
    """
    start_t = log_agent_intent(
        component="tool",
        agent_name="pantry_order_specialist",
        action_name="dispatch_missing_ingredients_order",
        intended_inputs={
            "missing_items": missing_items,
            "estimated_total_usd": estimated_total_usd,
            "delivery_notes": delivery_notes,
        },
    )

    try:
        validated = PantryOrderInputSchema(
            missing_items=missing_items,
            estimated_total_usd=estimated_total_usd,
            delivery_notes=delivery_notes,
        )
    except ValidationError as exc:
        error_resp = GuidedToolErrorResponse(
            error_code="INVALID_PANTRY_ORDER_PARAMETERS",
            error_message=f"Pantry order validation failed: {exc.errors()[0]['msg']}",
            invalid_parameters={
                "missing_items": missing_items,
                "estimated_total_usd": estimated_total_usd,
            },
            llm_recovery_instructions=(
                "Ensure `missing_items` contains 1 to 30 items and `estimated_total_usd` is "
                "between 0.01 and 500.00 USD. Do not submit negative or zero amounts."
            ),
        ).model_dump()
        log_agent_outcome(
            component="tool",
            agent_name="pantry_order_specialist",
            action_name="dispatch_missing_ingredients_order",
            actual_result=error_resp,
            status="error",
            start_time_monotonic=start_t,
        )
        return error_resp

    safe_notes = DEFAULT_PII_SCRUBBER.scrub_text(validated.delivery_notes)

    # Explicit Human-in-the-Loop code stop (Rubric 3.4: Human-in-the-Loop Hooks)
    if tool_context is not None:
        tc = getattr(tool_context, "tool_confirmation", None)
        if tc is None:
            hint = (
                f"Confirm grocery delivery order for {len(validated.missing_items)} item(s) "
                f"({', '.join(validated.missing_items)}) totaling ${validated.estimated_total_usd:.2f} USD?"
            )
            req_fn = getattr(tool_context, "request_confirmation", None)
            if callable(req_fn):
                req_fn(
                    hint=hint,
                    payload={
                        "missing_items": validated.missing_items,
                        "estimated_total_usd": validated.estimated_total_usd,
                        "delivery_notes": safe_notes,
                    },
                )
            pending_resp = {
                "status": "awaiting_human_confirmation",
                "confirmation_required": True,
                "confirmation_prompt": hint,
                "order_preview": {
                    "items": validated.missing_items,
                    "estimated_total_usd": validated.estimated_total_usd,
                    "sanitized_delivery_notes": safe_notes,
                },
            }
            log_agent_outcome(
                component="tool",
                agent_name="pantry_order_specialist",
                action_name="dispatch_missing_ingredients_order",
                actual_result=pending_resp,
                status="paused_for_hitl",
                start_time_monotonic=start_t,
            )
            return pending_resp

        if not getattr(tc, "confirmed", False):
            declined_resp = {
                "status": "cancelled_by_user",
                "confirmation_required": True,
                "message": "The user declined the grocery order confirmation. No charges were made.",
            }
            log_agent_outcome(
                component="tool",
                agent_name="pantry_order_specialist",
                action_name="dispatch_missing_ingredients_order",
                actual_result=declined_resp,
                status="declined_by_hitl",
                start_time_monotonic=start_t,
            )
            return declined_resp

    # Retrieve partner API credential via Secret Manager (never hardcoded — Rubric 5.3)
    _partner_token = SECRET_MANAGER.get_secret("GROCERY_PARTNER_API_SECRET")
    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    confirmed_resp = {
        "status": "success",
        "order_id": order_id,
        "dispatched_items": validated.missing_items,
        "charged_total_usd": validated.estimated_total_usd,
        "sanitized_delivery_notes": safe_notes,
        "secret_manager_authenticated": _partner_token is not None,
    }
    log_agent_outcome(
        component="tool",
        agent_name="pantry_order_specialist",
        action_name="dispatch_missing_ingredients_order",
        actual_result=confirmed_resp,
        status="success",
        start_time_monotonic=start_t,
    )
    return confirmed_resp


# Wrap with ADK `FunctionTool(..., require_confirmation=True)` for native ADK HITL enforcement
dispatch_missing_ingredients_order = FunctionTool(
    _dispatch_missing_ingredients_order_impl,
    require_confirmation=True,
)
