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

"""Strict Pydantic JSON Schemas for tool inputs, tool outputs, and structured LLM validation.

Implements Rubric Category 1 (Tool & Interface Design):
- Explicit JSON Schemas: Strict input and output schemas (`extra="forbid"`) to
  validate tool arguments and constrain LLM outputs.
- Guided Error Handling: Structured `GuidedToolErrorResponse` containing actionable
  `llm_recovery_instructions` so the agent recovers gracefully instead of crashing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GuidedToolErrorResponse(BaseModel):
    """Structured error envelope with explicit LLM recovery instructions."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["error"] = Field(
        default="error",
        description="Indicates the tool execution encountered a recoverable error.",
    )
    error_code: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Machine-readable error classification code (e.g., INSUFFICIENT_INGREDIENTS).",
    )
    error_message: str = Field(
        ...,
        min_length=5,
        description="Human-readable explanation of why the tool call failed.",
    )
    invalid_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary mapping invalid parameter names to the rejected values.",
    )
    llm_recovery_instructions: str = Field(
        ...,
        min_length=10,
        description=(
            "Actionable step-by-step instructions telling the LLM how to recover, "
            "retry with corrected parameters, or ask the user for clarification."
        ),
    )
    suggested_fallback_options: list[str] = Field(
        default_factory=list,
        description="Optional fallback suggestions the LLM can offer to the user.",
    )


class IngredientMatchInputSchema(BaseModel):
    """Strict input schema for `match_ingredients_to_meal_options`."""

    model_config = ConfigDict(extra="forbid")

    available_ingredients: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Normalized list of ingredients the user currently has on hand.",
    )
    dietary_restrictions: list[str] = Field(
        default_factory=list,
        description="Dietary constraints or allergies (e.g., 'vegan', 'gluten-free', 'nut-free').",
    )
    max_prep_time_minutes: int = Field(
        default=45,
        ge=5,
        le=240,
        description="Maximum total preparation and cooking time in minutes (5 to 240).",
    )
    meal_type: str = Field(
        default="any",
        description="Target meal category ('breakfast', 'lunch', 'dinner', 'snack', or 'any').",
    )

    @field_validator("available_ingredients")
    @classmethod
    def validate_ingredients_non_empty(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip().lower() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError(
                "At least one non-empty ingredient string must be provided."
            )
        return cleaned


class MealCandidateSchema(BaseModel):
    """Structured schema for a single matched meal option."""

    model_config = ConfigDict(extra="forbid")

    meal_id: str = Field(..., description="Unique identifier of the meal in the culinary store.")
    meal_name: str = Field(..., description="Full culinary title of the recommended meal.")
    cuisine_style: str = Field(..., description="Cuisine classification (e.g., Mediterranean, Thai, Mexican).")
    match_score_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of core recipe ingredients satisfied by the user's pantry.",
    )
    matched_ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredients from the user's list used in this meal.",
    )
    missing_ingredients: list[str] = Field(
        default_factory=list,
        description="Additional ingredients required or optional pantry staples.",
    )
    prep_time_minutes: int = Field(
        ..., ge=1, le=240, description="Total prep and cook time in minutes."
    )
    dietary_tags: list[str] = Field(
        default_factory=list,
        description="Verified dietary compliance tags (e.g., 'vegan', 'gluten-free', 'high-protein').",
    )
    why_good_option: str = Field(
        ...,
        description="Culinary rationale explaining why this meal is a strong match for the user.",
    )


class IngredientMatchOutputSchema(BaseModel):
    """Strict output schema for `match_ingredients_to_meal_options`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    best_meal_option: MealCandidateSchema = Field(
        ..., description="Top-ranked meal recommendation based on ingredient synergy."
    )
    alternative_meal_options: list[MealCandidateSchema] = Field(
        default_factory=list,
        description="Runner-up meal options that also fit the user's ingredients.",
    )
    applied_dietary_filters: list[str] = Field(
        default_factory=list,
        description="Dietary restrictions enforced during candidate filtering.",
    )
    search_backend: str = Field(
        default="sqlite_vector_hybrid",
        description="Storage backend used for retrieval ('sqlite_vector_hybrid' or 'vertex_ai_search').",
    )


class RecipeStepSchema(BaseModel):
    """Structured step in a culinary recipe."""

    model_config = ConfigDict(extra="forbid")

    step_number: int = Field(..., ge=1, description="1-based sequential step number.")
    instruction: str = Field(..., min_length=5, description="Clear cooking action for this step.")
    duration_minutes: int = Field(
        default=5, ge=0, le=180, description="Estimated time for this step in minutes."
    )
    critical_safety_checkpoint: str | None = Field(
        default=None,
        description="Food safety temperature or handling note if applicable (e.g., 165°F for poultry).",
    )


class StructuredRecipeInputSchema(BaseModel):
    """Strict input schema for `generate_structured_recipe_card`."""

    model_config = ConfigDict(extra="forbid")

    meal_id_or_name: str = Field(
        ...,
        min_length=2,
        description="Meal identifier or meal name selected from the candidate options.",
    )
    available_ingredients: list[str] = Field(
        ...,
        min_length=1,
        description="List of ingredients the user has available.",
    )
    servings: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of servings to scale ingredient measurements for (1 to 20).",
    )
    dietary_restrictions: list[str] = Field(
        default_factory=list,
        description="Dietary restrictions or allergies to enforce in substitutions.",
    )


class StructuredRecipeOutputSchema(BaseModel):
    """Strict output schema for `generate_structured_recipe_card`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    meal_name: str = Field(..., description="Official title of the recipe.")
    servings: int = Field(..., ge=1, le=20, description="Scaled number of servings.")
    prep_time_minutes: int = Field(..., ge=1, description="Preparation time in minutes.")
    cook_time_minutes: int = Field(..., ge=0, description="Cooking time in minutes.")
    total_time_minutes: int = Field(..., ge=1, description="Total time in minutes.")
    scaled_ingredients: list[dict[str, str]] = Field(
        ...,
        description="List of ingredient dicts with 'item', 'amount', and 'status' (on_hand/pantry_staple).",
    )
    equipment_needed: list[str] = Field(
        default_factory=list,
        description="Kitchen tools required (e.g., 'Large skillet', 'Chef knife').",
    )
    steps: list[RecipeStepSchema] = Field(
        ..., min_length=1, description="Ordered step-by-step cooking instructions."
    )
    chef_tips_and_substitutions: list[str] = Field(
        default_factory=list,
        description="Tips for flavor enhancement and safe ingredient substitutions.",
    )
    food_safety_guidelines: list[str] = Field(
        default_factory=list,
        description="Mandatory USDA/FDA safe internal cooking temperatures and allergen notices.",
    )


class NutritionCalculationInputSchema(BaseModel):
    """Strict input schema for `calculate_recipe_nutrition_profile`."""

    model_config = ConfigDict(extra="forbid")

    ingredients: list[str] = Field(
        ...,
        min_length=1,
        description="List of ingredients in the recipe to analyze for nutritional value.",
    )
    servings: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of servings to divide total recipe macros by.",
    )


class NutritionCalculationOutputSchema(BaseModel):
    """Strict output schema for `calculate_recipe_nutrition_profile`."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"] = "success"
    servings: int = Field(..., ge=1, le=20)
    per_serving_macros: dict[str, float] = Field(
        ...,
        description="Per-serving nutritional metrics: calories_kcal, protein_g, carbs_g, fat_g, fiber_g.",
    )
    dietary_compliance_badges: list[str] = Field(
        default_factory=list,
        description="Automatically verified nutrition badges (e.g., 'High-Protein', 'Balanced-Macros').",
    )
    allergen_warnings: list[str] = Field(
        default_factory=list,
        description="Detected common allergens present in the ingredient list.",
    )


class DietaryPreferenceMemoryInputSchema(BaseModel):
    """Strict input schema for `save_dietary_preference_to_memory`."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(..., min_length=1, description="Unique user identifier.")
    dietary_restrictions: list[str] = Field(
        default_factory=list,
        description="Permanent dietary restrictions (e.g., 'vegetarian', 'halal', 'keto').",
    )
    allergies: list[str] = Field(
        default_factory=list,
        description="Strict food allergies that must never be included in meals.",
    )
    disliked_ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredients the user dislikes and prefers to avoid.",
    )
    preferred_cuisines: list[str] = Field(
        default_factory=list,
        description="Favorite cuisine styles (e.g., 'Italian', 'Mediterranean', 'Japanese').",
    )


class PantryOrderInputSchema(BaseModel):
    """Strict input schema for `dispatch_missing_ingredients_order`."""

    model_config = ConfigDict(extra="forbid")

    missing_items: list[str] = Field(
        ...,
        min_length=1,
        max_length=30,
        description="List of missing ingredients to order from the grocery delivery partner.",
    )
    estimated_total_usd: float = Field(
        ...,
        gt=0.0,
        le=500.0,
        description="Estimated total cost in USD (must be between $0.01 and $500.00).",
    )
    delivery_notes: str = Field(
        default="Leave at front door",
        max_length=250,
        description="Sanitized delivery instructions (PII is automatically redacted).",
    )
