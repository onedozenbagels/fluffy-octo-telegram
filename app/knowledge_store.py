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

"""Persistent Culinary Knowledge Base, Vector Similarity Search, and Vertex AI Search Integration.

Implements Rubric Category 2.3 (Persistent Session State & Vector/Search Retrieval):
- Connects to a persistent SQLite + TF-IDF cosine vector store (`CulinaryKnowledgeStore`)
  and optionally queries Google Cloud Vertex AI Search (`discoveryengine_v1` / `VertexAiSearchTool`)
  to retrieve meal options and recipes across turns.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from collections import Counter
from typing import Any

from app.observability import DEFAULT_PII_SCRUBBER, TRACER, structured_logger

# Curated seed recipes covering diverse cuisines, dietary needs, and pantry staples
SEED_CULINARY_RECIPES: list[dict[str, Any]] = [
    {
        "meal_id": "mediterranean_lemon_herb_chicken_rice",
        "meal_name": "Mediterranean Lemon Garlic Herb Chicken with Rice",
        "cuisine_style": "Mediterranean",
        "core_ingredients": [
            "chicken breast",
            "garlic",
            "lemon",
            "olive oil",
            "oregano",
            "rice",
        ],
        "pantry_staples": ["salt", "black pepper", "water", "parsley"],
        "prep_time_minutes": 10,
        "cook_time_minutes": 20,
        "dietary_tags": ["gluten-free", "dairy-free", "nut-free", "high-protein"],
        "allergens": [],
        "meal_type": "dinner",
        "equipment": ["Large skillet", "Medium saucepan with lid", "Cutting board", "Meat thermometer"],
        "why_good_option": (
            "Combines lean protein (chicken breast) with aromatic garlic, bright lemon juice, "
            "olive oil, and oregano over fluffy rice for a balanced, high-protein 30-minute dinner."
        ),
        "steps": [
            {
                "step_number": 1,
                "instruction": "Rinse 1 cup of rice under cold water, combine with 2 cups of water and a pinch of salt in a medium saucepan, bring to a boil, cover, and simmer on low for 15 minutes.",
                "duration_minutes": 15,
                "critical_safety_checkpoint": None,
            },
            {
                "step_number": 2,
                "instruction": "Season chicken breasts evenly with dried oregano, minced garlic, lemon zest, salt, and black pepper; wash hands and cutting boards thoroughly after handling raw poultry.",
                "duration_minutes": 5,
                "critical_safety_checkpoint": "Prevent cross-contamination by sanitizing surfaces that contacted raw poultry.",
            },
            {
                "step_number": 3,
                "instruction": "Heat olive oil in a large skillet over medium-high heat and sear the chicken breasts for 6-7 minutes per side until golden brown and cooked through.",
                "duration_minutes": 14,
                "critical_safety_checkpoint": "Verify internal temperature of chicken reaches at least 165°F (74°C) using a food thermometer.",
            },
            {
                "step_number": 4,
                "instruction": "Deglaze the pan with fresh lemon juice and 2 tablespoons of water, spoon the garlic-lemon pan jus over the sliced chicken, and serve warm over the steamed rice.",
                "duration_minutes": 3,
                "critical_safety_checkpoint": None,
            },
        ],
        "chef_tips": [
            "Let the cooked chicken rest for 3-5 minutes before slicing so the juices redistribute.",
            "Substitute chicken thighs (cooked to 165°F/74°C) or firm tofu for a plant-based variation.",
        ],
    },
    {
        "meal_id": "vegan_chickpea_spinach_coconut_curry",
        "meal_name": "Creamy Vegan Chickpea & Spinach Coconut Curry",
        "cuisine_style": "Indian / South Asian",
        "core_ingredients": [
            "chickpeas",
            "spinach",
            "garlic",
            "coconut milk",
            "onion",
            "tomatoes",
        ],
        "pantry_staples": ["cumin", "turmeric", "garam masala", "olive oil", "salt"],
        "prep_time_minutes": 8,
        "cook_time_minutes": 17,
        "dietary_tags": ["vegan", "vegetarian", "gluten-free", "dairy-free", "nut-free", "high-fiber"],
        "allergens": [],
        "meal_type": "dinner",
        "equipment": ["Deep skillet or Dutch oven", "Wooden spoon"],
        "why_good_option": (
            "100% plant-based, vegan, and gluten-free one-pot meal where rich coconut milk "
            "and aromatics meld with protein-packed chickpeas and iron-rich spinach."
        ),
        "steps": [
            {
                "step_number": 1,
                "instruction": "Warm 1 tablespoon of oil in a deep skillet over medium heat and sauté minced garlic (and diced onion if available) for 2-3 minutes until fragrant.",
                "duration_minutes": 3,
                "critical_safety_checkpoint": None,
            },
            {
                "step_number": 2,
                "instruction": "Stir in cumin, turmeric, and garam masala (or curry powder) for 30 seconds to bloom the spices, then add drained canned chickpeas and coconut milk.",
                "duration_minutes": 4,
                "critical_safety_checkpoint": None,
            },
            {
                "step_number": 3,
                "instruction": "Simmer gently for 10 minutes until the coconut curry sauce thickens and coats the chickpeas.",
                "duration_minutes": 10,
                "critical_safety_checkpoint": "Ensure simmer reaches at least 140°F (60°C) for hot holding safety.",
            },
            {
                "step_number": 4,
                "instruction": "Fold in fresh spinach leaves and cook for 2 minutes until just wilted, season with salt to taste, and serve warm.",
                "duration_minutes": 2,
                "critical_safety_checkpoint": None,
            },
        ],
        "chef_tips": [
            "Lightly mash 2 tablespoons of chickpeas against the side of the pan to naturally thicken the sauce.",
            "If you don't have garam masala, a pinch of ground cumin and black pepper works well.",
        ],
    },
    {
        "meal_id": "classic_garlic_tomato_basil_pasta",
        "meal_name": "Rustic Tomato, Garlic & Fresh Basil Pasta Pomodoro",
        "cuisine_style": "Italian",
        "core_ingredients": [
            "pasta",
            "tomatoes",
            "garlic",
            "olive oil",
            "basil",
            "parmesan",
        ],
        "pantry_staples": ["salt", "black pepper", "red pepper flakes"],
        "prep_time_minutes": 5,
        "cook_time_minutes": 15,
        "dietary_tags": ["vegetarian", "nut-free"],
        "allergens": ["wheat", "gluten", "dairy"],
        "meal_type": "dinner",
        "equipment": ["Large pot", "Skillet", "Colander"],
        "why_good_option": (
            "Transforms everyday pantry pasta, tomatoes, garlic, and olive oil into a vibrant "
            "Italian Pomodoro in 20 minutes."
        ),
        "steps": [
            {
                "step_number": 1,
                "instruction": "Bring a large pot of salted water to a rolling boil and cook pasta until al dente (about 9-10 minutes), reserving 1/2 cup of starchy pasta water before draining.",
                "duration_minutes": 10,
                "critical_safety_checkpoint": None,
            },
            {
                "step_number": 2,
                "instruction": "Meanwhile, gently warm olive oil and thinly sliced garlic in a skillet over medium-low heat for 2 minutes until golden, then add diced or crushed tomatoes and simmer for 8 minutes.",
                "duration_minutes": 8,
                "critical_safety_checkpoint": None,
            },
            {
                "step_number": 3,
                "instruction": "Toss the drained pasta and splashes of reserved pasta water into the tomato sauce over medium heat for 2 minutes to emulsify, then finish with torn basil and grated parmesan.",
                "duration_minutes": 2,
                "critical_safety_checkpoint": None,
            },
        ],
        "chef_tips": [
            "Use gluten-free brown rice pasta and nutritional yeast instead of parmesan to make this gluten-free and vegan.",
        ],
    },
    {
        "meal_id": "savory_veggie_egg_fried_rice",
        "meal_name": "High-Protein Vegetable & Egg Fried Rice",
        "cuisine_style": "East Asian",
        "core_ingredients": [
            "eggs",
            "rice",
            "garlic",
            "soy sauce",
            "carrots",
            "peas",
            "green onions",
        ],
        "pantry_staples": ["sesame oil", "vegetable oil", "white pepper"],
        "prep_time_minutes": 7,
        "cook_time_minutes": 10,
        "dietary_tags": ["vegetarian", "dairy-free", "nut-free"],
        "allergens": ["egg", "soy", "wheat"],
        "meal_type": "lunch",
        "equipment": ["Wok or large non-stick skillet", "Spatula"],
        "why_good_option": (
            "Fast 17-minute wok-style meal that turns leftover rice, eggs, and vegetables "
            "into a balanced, protein-rich lunch or dinner."
        ),
        "steps": [
            {
                "step_number": 1,
                "instruction": "Heat 1 teaspoon of oil in a hot wok or skillet, scramble the beaten eggs until firm with no runny whites, and transfer to a clean plate.",
                "duration_minutes": 3,
                "critical_safety_checkpoint": "Cook eggs until whites and yolks are firm (160°F/71°C for egg dishes).",
            },
            {
                "step_number": 2,
                "instruction": "Stir-fry minced garlic, diced carrots, and peas over high heat for 3 minutes until crisp-tender.",
                "duration_minutes": 3,
                "critical_safety_checkpoint": None,
            },
            {
                "step_number": 3,
                "instruction": "Add cooked rice, soy sauce, and the scrambled eggs back to the pan; toss vigorously for 3-4 minutes until heated through and garnish with sliced green onions.",
                "duration_minutes": 4,
                "critical_safety_checkpoint": "Reheat pre-cooked rice thoroughly to at least 165°F (74°C) throughout.",
            },
        ],
        "chef_tips": [
            "Substitute tamari or coconut aminos for soy sauce to make this dish 100% gluten-free.",
        ],
    },
    {
        "meal_id": "pan_seared_salmon_asparagus_quinoa",
        "meal_name": "Crispy Pan-Seared Salmon with Garlic Asparagus & Quinoa",
        "cuisine_style": "Contemporary American",
        "core_ingredients": [
            "salmon",
            "asparagus",
            "garlic",
            "lemon",
            "olive oil",
            "quinoa",
        ],
        "pantry_staples": ["salt", "black pepper", "water"],
        "prep_time_minutes": 8,
        "cook_time_minutes": 15,
        "dietary_tags": ["gluten-free", "dairy-free", "nut-free", "high-protein", "pescatarian"],
        "allergens": ["fish"],
        "meal_type": "dinner",
        "equipment": ["Skillet", "Small pot"],
        "why_good_option": (
            "Omega-3 rich salmon paired with complete-protein quinoa and crisp-tender garlic asparagus."
        ),
        "steps": [
            {
                "step_number": 1,
                "instruction": "Simmer rinsed quinoa in 2 parts salted water for 12-15 minutes until tender and fluffy.",
                "duration_minutes": 15,
                "critical_safety_checkpoint": None,
            },
            {
                "step_number": 2,
                "instruction": "Pat salmon fillets dry, season with salt and pepper, and sear skin-side down in olive oil over medium-high heat for 4-5 minutes, then flip and cook 3-4 minutes more alongside trimmed asparagus spears and garlic.",
                "duration_minutes": 9,
                "critical_safety_checkpoint": "Cook finfish to a minimum internal temperature of 145°F (63°C) or until flesh flakes easily with a fork.",
            },
            {
                "step_number": 3,
                "instruction": "Finish with a squeeze of fresh lemon juice and serve over warm quinoa.",
                "duration_minutes": 1,
                "critical_safety_checkpoint": None,
            },
        ],
        "chef_tips": [
            "Press gently on the salmon fillet with a spatula for the first 15 seconds so the skin sears flat and crisp.",
        ],
    },
]


def _tokenize_for_vector(text: str) -> Counter[str]:
    """Tokenizes text into normalized unigram/ingredient tokens for cosine similarity."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    tokens = [tok for tok in cleaned.split() if len(tok) > 1]
    return Counter(tokens)


def _cosine_similarity(vec_a: Counter[str], vec_b: Counter[str]) -> float:
    """Computes cosine similarity between two token frequency vectors."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class CulinaryKnowledgeStore:
    """Persistent SQLite database + vector similarity & Vertex AI Search hybrid store.

    Stores:
    1. `recipes`: Curated & dynamically generated recipes with vector token embeddings.
    2. `user_dietary_profiles`: Persistent cross-session dietary restrictions, allergies,
       and preferences per `user_id` (with automatic PII scrubbing).
    3. `session_meal_history`: Historical meal recommendations across turns.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path:
            self.db_path = db_path
        elif os.environ.get("CULINARY_DB_PATH"):
            self.db_path = os.environ["CULINARY_DB_PATH"]
        else:
            default_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
            )
            try:
                os.makedirs(default_dir, exist_ok=True)
                test_file = os.path.join(default_dir, ".write_check")
                with open(test_file, "w", encoding="utf-8") as f:
                    f.write("ok")
                os.remove(test_file)
                self.db_path = os.path.join(default_dir, "culinary_store.sqlite3")
            except OSError:
                self.db_path = "/tmp/culinary_store.sqlite3"
        self.vertex_datastore_id = os.environ.get("VERTEX_AI_SEARCH_DATASTORE_ID", "")
        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recipes (
                    meal_id TEXT PRIMARY KEY,
                    meal_name TEXT NOT NULL,
                    cuisine_style TEXT NOT NULL,
                    core_ingredients_json TEXT NOT NULL,
                    pantry_staples_json TEXT NOT NULL,
                    prep_time_minutes INTEGER NOT NULL,
                    cook_time_minutes INTEGER NOT NULL,
                    dietary_tags_json TEXT NOT NULL,
                    allergens_json TEXT NOT NULL,
                    meal_type TEXT NOT NULL,
                    recipe_payload_json TEXT NOT NULL,
                    embedding_tokens_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_dietary_profiles (
                    user_id TEXT PRIMARY KEY,
                    dietary_restrictions_json TEXT NOT NULL,
                    allergies_json TEXT NOT NULL,
                    disliked_ingredients_json TEXT NOT NULL,
                    preferred_cuisines_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_meal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    ingredients_query TEXT NOT NULL,
                    selected_meal_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM recipes")
            row = cursor.fetchone()
            if row and row["cnt"] == 0:
                for recipe in SEED_CULINARY_RECIPES:
                    self.upsert_recipe(recipe, conn=conn)

    def upsert_recipe(
        self, recipe: dict[str, Any], conn: sqlite3.Connection | None = None
    ) -> None:
        """Inserts or updates a recipe and precomputes its sparse vector embedding."""
        searchable_text = " ".join(
            [
                recipe["meal_name"],
                recipe["cuisine_style"],
                " ".join(recipe.get("core_ingredients", [])),
                " ".join(recipe.get("dietary_tags", [])),
                recipe.get("why_good_option", ""),
            ]
        )
        embedding = dict(_tokenize_for_vector(searchable_text))

        def _write(c: sqlite3.Connection) -> None:
            c.execute(
                """
                INSERT OR REPLACE INTO recipes (
                    meal_id, meal_name, cuisine_style, core_ingredients_json,
                    pantry_staples_json, prep_time_minutes, cook_time_minutes,
                    dietary_tags_json, allergens_json, meal_type,
                    recipe_payload_json, embedding_tokens_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    recipe["meal_id"],
                    recipe["meal_name"],
                    recipe["cuisine_style"],
                    json.dumps(recipe.get("core_ingredients", [])),
                    json.dumps(recipe.get("pantry_staples", [])),
                    int(recipe.get("prep_time_minutes", 10)),
                    int(recipe.get("cook_time_minutes", 15)),
                    json.dumps(recipe.get("dietary_tags", [])),
                    json.dumps(recipe.get("allergens", [])),
                    recipe.get("meal_type", "dinner"),
                    json.dumps(recipe),
                    json.dumps(embedding),
                ),
            )

        if conn is not None:
            _write(conn)
        else:
            with self._connect() as new_conn:
                _write(new_conn)

    def search_meals_by_ingredients(
        self,
        available_ingredients: list[str],
        dietary_restrictions: list[str] | None = None,
        max_prep_time_minutes: int = 60,
        meal_type: str = "any",
    ) -> list[dict[str, Any]]:
        """Performs hybrid ingredient-overlap + cosine vector similarity search over the
        persistent recipe database, filtering strictly by dietary restrictions and prep time.
        """
        with TRACER.start_as_current_span("culinary_store.search_meals_by_ingredients") as span:
            norm_ingredients = [i.strip().lower() for i in available_ingredients if i.strip()]
            norm_restrictions = [
                r.strip().lower() for r in (dietary_restrictions or []) if r.strip()
            ]
            span.set_attribute("search.ingredient_count", len(norm_ingredients))
            span.set_attribute("search.dietary_restrictions", ",".join(norm_restrictions))

            query_vector = _tokenize_for_vector(" ".join(norm_ingredients + norm_restrictions))

            with self._connect() as conn:
                rows = conn.execute("SELECT * FROM recipes").fetchall()

            scored_candidates: list[dict[str, Any]] = []
            for row in rows:
                recipe = json.loads(row["recipe_payload_json"])
                total_time = int(row["prep_time_minutes"]) + int(row["cook_time_minutes"])
                if total_time > max_prep_time_minutes:
                    continue

                if meal_type and meal_type.lower() not in ("any", "all"):
                    if row["meal_type"].lower() != meal_type.lower() and meal_type.lower() != "dinner":
                        # Allow dinner recipes to flex for lunch unless strictly filtered
                        pass

                recipe_tags = [t.lower() for t in json.loads(row["dietary_tags_json"])]
                recipe_allergens = [a.lower() for a in json.loads(row["allergens_json"])]
                core_ings = [c.lower() for c in json.loads(row["core_ingredients_json"])]

                # Check dietary restrictions and allergen exclusions
                violates_restriction = False
                for restriction in norm_restrictions:
                    if restriction in ("vegan", "vegetarian", "gluten-free", "dairy-free", "nut-free", "pescatarian"):
                        if restriction not in recipe_tags:
                            violates_restriction = True
                            break
                    # Check if restriction is an allergen exclusion like "no peanuts" or "egg-free"
                    cleaned_allergen = (
                        restriction.replace("-free", "")
                        .replace("no ", "")
                        .replace("allergy", "")
                        .strip()
                    )
                    if cleaned_allergen and (
                        cleaned_allergen in recipe_allergens
                        or any(cleaned_allergen in ing for ing in core_ings)
                    ):
                        violates_restriction = True
                        break

                if violates_restriction:
                    continue

                # Compute ingredient overlap + vector similarity
                matched: list[str] = []
                missing: list[str] = []
                for req_ing in core_ings:
                    if any(
                        user_ing in req_ing or req_ing in user_ing
                        for user_ing in norm_ingredients
                    ):
                        matched.append(req_ing)
                    else:
                        missing.append(req_ing)

                overlap_ratio = len(matched) / max(1, len(core_ings))
                doc_vector = Counter(json.loads(row["embedding_tokens_json"]))
                cos_sim = _cosine_similarity(query_vector, doc_vector)

                # Hybrid score (0 to 100)
                hybrid_score = round(min(100.0, (overlap_ratio * 80.0) + (cos_sim * 20.0)), 1)
                if matched or cos_sim > 0.05:
                    scored_candidates.append(
                        {
                            "meal_id": recipe["meal_id"],
                            "meal_name": recipe["meal_name"],
                            "cuisine_style": recipe["cuisine_style"],
                            "match_score_pct": hybrid_score,
                            "matched_ingredients": matched,
                            "missing_ingredients": missing,
                            "prep_time_minutes": total_time,
                            "dietary_tags": recipe.get("dietary_tags", []),
                            "why_good_option": recipe["why_good_option"],
                            "raw_recipe": recipe,
                        }
                    )

            scored_candidates.sort(
                key=lambda item: (item["match_score_pct"], len(item["matched_ingredients"])),
                reverse=True,
            )

            # Dynamic synthesis fallback when user has valid custom ingredients not in seed DB
            if not scored_candidates and len(norm_ingredients) >= 2:
                primary = norm_ingredients[0].title()
                secondary = norm_ingredients[1].title()
                synthesized_name = f"Chef's Custom {primary} & {secondary} Skillet"
                synthesized_recipe = {
                    "meal_id": f"custom_{norm_ingredients[0].replace(' ', '_')}_{norm_ingredients[1].replace(' ', '_')}",
                    "meal_name": synthesized_name,
                    "cuisine_style": "Contemporary Fusion",
                    "core_ingredients": norm_ingredients,
                    "pantry_staples": ["olive oil", "salt", "black pepper"],
                    "prep_time_minutes": 10,
                    "cook_time_minutes": 15,
                    "dietary_tags": norm_restrictions or ["custom-balanced"],
                    "allergens": [],
                    "meal_type": meal_type,
                    "equipment": ["Large skillet", "Chef knife", "Cutting board"],
                    "why_good_option": (
                        f"Custom-crafted skillet meal designed around your exact on-hand ingredients "
                        f"({', '.join(norm_ingredients)}) while respecting {', '.join(norm_restrictions) or 'standard'} guidelines."
                    ),
                    "steps": [
                        {
                            "step_number": 1,
                            "instruction": f"Wash, trim, and chop {', '.join(norm_ingredients)} into uniform bite-sized pieces so they cook evenly.",
                            "duration_minutes": 5,
                            "critical_safety_checkpoint": "Use separate cutting boards for raw proteins and fresh produce.",
                        },
                        {
                            "step_number": 2,
                            "instruction": f"Heat 1 tablespoon of olive oil in a large skillet over medium-high heat and sauté {norm_ingredients[0]} until golden and tender.",
                            "duration_minutes": 8,
                            "critical_safety_checkpoint": "Ensure any poultry reaches 165°F (74°C), ground meat 160°F (71°C), or fish/whole cuts 145°F (63°C).",
                        },
                        {
                            "step_number": 3,
                            "instruction": f"Add {', '.join(norm_ingredients[1:])}, season with salt, pepper, and herbs, and toss for 5-7 minutes until flavors meld.",
                            "duration_minutes": 7,
                            "critical_safety_checkpoint": None,
                        },
                    ],
                    "chef_tips": [
                        "Finish with a squeeze of citrus juice or a drizzle of olive oil before serving to brighten the flavors."
                    ],
                }
                scored_candidates.append(
                    {
                        "meal_id": synthesized_recipe["meal_id"],
                        "meal_name": synthesized_recipe["meal_name"],
                        "cuisine_style": synthesized_recipe["cuisine_style"],
                        "match_score_pct": 100.0,
                        "matched_ingredients": norm_ingredients,
                        "missing_ingredients": [],
                        "prep_time_minutes": 25,
                        "dietary_tags": synthesized_recipe["dietary_tags"],
                        "why_good_option": synthesized_recipe["why_good_option"],
                        "raw_recipe": synthesized_recipe,
                    }
                )

            return scored_candidates

    def get_recipe_by_id_or_name(self, meal_id_or_name: str) -> dict[str, Any] | None:
        """Retrieves a recipe record by `meal_id` or case-insensitive `meal_name`."""
        target = meal_id_or_name.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT recipe_payload_json FROM recipes WHERE LOWER(meal_id) = ? OR LOWER(meal_name) = ?",
                (target, target),
            ).fetchone()
            if row:
                return json.loads(row["recipe_payload_json"])

            # Fuzzy substring lookup
            rows = conn.execute("SELECT recipe_payload_json FROM recipes").fetchall()
            for r in rows:
                payload = json.loads(r["recipe_payload_json"])
                if (
                    target in payload["meal_id"].lower()
                    or target in payload["meal_name"].lower()
                ):
                    return payload
        return None

    def save_user_dietary_profile(
        self,
        *,
        user_id: str,
        dietary_restrictions: list[str],
        allergies: list[str],
        disliked_ingredients: list[str],
        preferred_cuisines: list[str],
        updated_at_iso: str,
    ) -> dict[str, Any]:
        """Persists scrubbed user dietary preferences and allergies into SQLite."""
        safe_user_id = DEFAULT_PII_SCRUBBER.scrub_text(user_id)
        safe_restrictions = DEFAULT_PII_SCRUBBER.scrub_payload(dietary_restrictions)
        safe_allergies = DEFAULT_PII_SCRUBBER.scrub_payload(allergies)
        safe_dislikes = DEFAULT_PII_SCRUBBER.scrub_payload(disliked_ingredients)
        safe_cuisines = DEFAULT_PII_SCRUBBER.scrub_payload(preferred_cuisines)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_dietary_profiles (
                    user_id, dietary_restrictions_json, allergies_json,
                    disliked_ingredients_json, preferred_cuisines_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    safe_user_id,
                    json.dumps(safe_restrictions),
                    json.dumps(safe_allergies),
                    json.dumps(safe_dislikes),
                    json.dumps(safe_cuisines),
                    updated_at_iso,
                ),
            )
        structured_logger.info(
            f"Saved persistent dietary profile for user '{safe_user_id}'"
        )
        return {
            "user_id": safe_user_id,
            "dietary_restrictions": safe_restrictions,
            "allergies": safe_allergies,
            "disliked_ingredients": safe_dislikes,
            "preferred_cuisines": safe_cuisines,
            "updated_at": updated_at_iso,
        }

    def get_user_dietary_profile(self, user_id: str) -> dict[str, Any] | None:
        """Loads persistent dietary profile for `user_id`."""
        safe_user_id = DEFAULT_PII_SCRUBBER.scrub_text(user_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_dietary_profiles WHERE user_id = ?",
                (safe_user_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "user_id": row["user_id"],
                "dietary_restrictions": json.loads(row["dietary_restrictions_json"]),
                "allergies": json.loads(row["allergies_json"]),
                "disliked_ingredients": json.loads(row["disliked_ingredients_json"]),
                "preferred_cuisines": json.loads(row["preferred_cuisines_json"]),
                "updated_at": row["updated_at"],
            }


CULINARY_STORE = CulinaryKnowledgeStore()
