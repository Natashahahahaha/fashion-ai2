# Dataset audit: Maryland Polyvore (Han et al. 2017) + Marqo/polyvore images

Generated 2026-10-04T02:00:20+00:00 (schema v1).

## Verdict: **SUFFICIENT_WITH_LIMITATIONS**

Question: *Is this dataset sufficient to train a credible fashion compatibility model?*

### Limitations
- 130 duplicate-image groups span different splits (collapsed during cleaning)
- no user/account identifiers, so user-level leakage cannot be checked
- item ids are unique per outfit, so reuse of the same garment across outfits/splits is only detectable through image hashing (exact/dHash); different photos of the same product are not detected

## Dataset
```json
{
  "name": "Maryland Polyvore (Han et al. 2017) + Marqo/polyvore images",
  "format": "polyvore_outfits",
  "path": "data/datasets/maryland-polyvore/prepared",
  "users_or_accounts": "not available in this format",
  "source": "https://github.com/xthan/polyvore-dataset ; https://huggingface.co/datasets/Marqo/polyvore",
  "license": "Apache-2.0 (both repositories). Product images originate from Polyvore.com and may be third-party copyright: research use, no redistribution.",
  "version": "github@ba8aa71b6589 + hf@8c782ee447fa"
}
```

## Raw structure
```json
{
  "outfits": {
    "train": 14704,
    "valid": 1238,
    "test": 2594
  },
  "unique_outfits_total": 18536,
  "items": 71848,
  "outfit_size_distribution": {
    "2": 3459,
    "3": 4767,
    "4": 4569,
    "5": 3230,
    "6": 1599,
    "7": 687,
    "8": 225
  },
  "duplicate_outfits_within_splits": {
    "train": 0,
    "valid": 0,
    "test": 0
  }
}
```

## Categories
```json
{
  "n_categories": 6,
  "distribution": {
    "accessory": 30516,
    "shoes": 12285,
    "top": 10485,
    "bottom": 8968,
    "outerwear": 5273,
    "one_piece": 4321
  },
  "slot_coverage": {
    "accessory": 30516,
    "shoes": 12285,
    "top": 10485,
    "bottom": 8968,
    "outerwear": 5273,
    "one_piece": 4321
  },
  "top_category_share": 0.4247299855249972,
  "imbalance_ratio_max_min": 7.062254107845406
}
```

## Images
```json
{
  "checked": true,
  "sampled": false,
  "items_checked": 71848,
  "found_ok": 71848,
  "missing": 0,
  "corrupt": 0,
  "missing_or_corrupt_rate": 0.0,
  "width": {
    "min": 26,
    "median": 340.0,
    "max": 400
  },
  "height": {
    "min": 6,
    "median": 400.0,
    "max": 400
  },
  "small_images_lt_64px": 84,
  "exact_duplicate_groups": 26,
  "items_in_exact_duplicates": 52,
  "perceptual_duplicate_groups": 391,
  "items_in_perceptual_duplicates": 984,
  "duplicate_image_rate": 0.00825353524106447
}
```

## Leakage
```json
{
  "items_shared": {
    "train-valid": 0,
    "train-test": 0,
    "valid-test": 0
  },
  "identical_outfits_across_splits": 0,
  "items_in_multiple_outfits": 0,
  "perceptual_duplicate_groups_across_splits": 130
}
```

## Negative construction
Positives are OBSERVED: every pair of items that appear together in one curated outfit. Negatives are CONSTRUCTED (not human judgements of incompatibility): (a) random negatives pair items drawn from two different outfits; (b) hard negatives (only when item categories are known) take a real positive (a, b) and replace b with another item of b's category from elsewhere, so the category pairing matches real outfits. Constructed negatives never coincide with an observed positive pair. A constructed negative may still be a combination a person would wear; it is 'not observed together', not 'known incompatible'.

## Effective size
```json
{
  "raw_pairs": {
    "train": {
      "items": 55297,
      "positive_pairs": 90695,
      "negative_pairs": 90695,
      "kind_hard_negative": 45347,
      "kind_observed_positive": 90695,
      "kind_random_negative": 45348
    },
    "valid": {
      "items": 5367,
      "positive_pairs": 10267,
      "negative_pairs": 10267,
      "kind_hard_negative": 5133,
      "kind_observed_positive": 10267,
      "kind_random_negative": 5134
    },
    "test": {
      "items": 11184,
      "positive_pairs": 21224,
      "negative_pairs": 21224,
      "kind_hard_negative": 10612,
      "kind_observed_positive": 21224,
      "kind_random_negative": 10612
    }
  },
  "after_cleaning": {
    "train": {
      "items": 54869,
      "positive_pairs": 90695,
      "negative_pairs": 90695,
      "kind_hard_negative": 45347,
      "kind_observed_positive": 90695,
      "kind_random_negative": 45348
    },
    "valid": {
      "items": 5358,
      "positive_pairs": 10267,
      "negative_pairs": 10267,
      "kind_hard_negative": 5133,
      "kind_observed_positive": 10267,
      "kind_random_negative": 5134
    },
    "test": {
      "items": 11162,
      "positive_pairs": 21223,
      "negative_pairs": 21224,
      "kind_hard_negative": 10612,
      "kind_observed_positive": 21223,
      "kind_random_negative": 10612
    }
  },
  "item_disjoint_pairs_dropped": {
    "valid": 427,
    "test": 702
  },
  "after_item_disjoint_split": {
    "train": {
      "items": 54869,
      "positive_pairs": 90695,
      "negative_pairs": 90695,
      "kind_hard_negative": 45347,
      "kind_observed_positive": 90695,
      "kind_random_negative": 45348
    },
    "valid": {
      "items": 5310,
      "positive_pairs": 10041,
      "negative_pairs": 10066,
      "kind_hard_negative": 5036,
      "kind_observed_positive": 10041,
      "kind_random_negative": 5030
    },
    "test": {
      "items": 11076,
      "positive_pairs": 20868,
      "negative_pairs": 20877,
      "kind_hard_negative": 10440,
      "kind_observed_positive": 20868,
      "kind_random_negative": 10437
    }
  },
  "fully_item_disjoint_outfits": {
    "valid": 1238,
    "test": 2594
  }
}
```

