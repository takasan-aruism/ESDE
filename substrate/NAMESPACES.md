# ESDE Substrate Layer: Namespace Registry

## Overview

This document defines the official namespaces for Substrate traces.
New namespaces must be added via PR and reviewed to prevent uncontrolled growth.

**Philosophy:** "Describe, but do not decide."

**Last Updated:** 2026-01-24  
**Version:** v0.1.0

---

## Rules

### 1. Naming Convention

- Namespace: lowercase letters only (`[a-z]+`)
- Key format: `namespace:name` (e.g., `html:tag_count`)
- Name: lowercase letters, numbers, underscores (`[a-z][a-z0-9_]*`)

### 2. Forbidden Namespaces (INV-SUB-002)

The following namespaces are **PERMANENTLY BANNED** because they imply semantic interpretation:

| Namespace | Reason |
|-----------|--------|
| `meaning:` | Semantic interpretation |
| `category:` | Classification |
| `intent:` | Intent inference |
| `quality:` | Quality judgment |
| `importance:` | Importance ranking |
| `sentiment:` | Sentiment analysis |
| `topic:` | Topic classification |
| `type:` | Type classification (use `meta:` instead) |

### 3. Adding New Namespaces

1. Create a PR with the new namespace added to this file
2. Provide justification (what machine-observable facts it captures)
3. Ensure no semantic interpretation is implied
4. Get review approval from at least one other contributor

---

## Official Namespaces

### `html:` - HTML Structure

Machine-observable facts about HTML document structure.

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `html:tag_count` | int | Total number of HTML tags | `42` |
| `html:has_h1` | bool | Document contains h1 tag | `true` |
| `html:has_h2` | bool | Document contains h2 tag | `true` |
| `html:has_script` | bool | Document contains script tag | `false` |
| `html:has_style` | bool | Document contains style tag | `true` |
| `html:has_form` | bool | Document contains form tag | `false` |
| `html:has_table` | bool | Document contains table tag | `false` |
| `html:has_img` | bool | Document contains img tag | `true` |
| `html:has_a` | bool | Document contains anchor tag | `true` |
| `html:max_depth` | int | Maximum nesting depth | `8` |
| `html:doctype` | str | DOCTYPE declaration | `"html"` |

---

### `text:` - Text Statistics

Machine-computable statistics about text content.

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `text:char_count` | int | Total character count | `1234` |
| `text:word_count` | int | Total word count (whitespace split) | `256` |
| `text:line_count` | int | Total line count | `42` |
| `text:sentence_count` | int | Estimated sentence count | `15` |
| `text:avg_word_len` | float | Average word length | `4.5` |
| `text:avg_sentence_len` | float | Average sentence length (words) | `17.1` |
| `text:has_punctuation` | bool | Contains sentence-ending punctuation | `true` |
| `text:has_numbers` | bool | Contains numeric characters | `true` |
| `text:whitespace_ratio` | float | Ratio of whitespace to total chars | `0.15` |

---

### `meta:` - Retrieval Metadata

Machine-observable facts about how the content was retrieved.

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `meta:domain` | str | Domain name from URL | `"example.com"` |
| `meta:scheme` | str | URL scheme | `"https"` |
| `meta:path_depth` | int | URL path depth | `3` |
| `meta:has_query` | bool | URL has query string | `true` |
| `meta:content_type` | str | HTTP Content-Type header | `"text/html"` |
| `meta:charset` | str | Character encoding | `"utf-8"` |
| `meta:content_length` | int | HTTP Content-Length header | `12345` |

---

### `time:` - Temporal Information

Machine-observable temporal facts.

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `time:year` | int | Year of observation | `2026` |
| `time:month` | int | Month of observation (1-12) | `1` |
| `time:day` | int | Day of month (1-31) | `24` |
| `time:hour` | int | Hour of day (0-23) | `14` |
| `time:weekday` | int | Day of week (0=Mon, 6=Sun) | `4` |
| `time:timestamp` | str | ISO8601 timestamp | `"2026-01-24T14:30:00Z"` |

---

### `struct:` - Structural Patterns

Machine-detectable structural patterns.

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `struct:reply_depth` | int | Nesting depth in reply chain | `3` |
| `struct:has_quoted_text` | bool | Contains quoted text markers | `true` |
| `struct:has_list` | bool | Contains list structures | `true` |
| `struct:has_code_block` | bool | Contains code block markers | `false` |
| `struct:paragraph_count` | int | Number of paragraphs | `5` |

---

### `env:` - Environment Information

Machine-observable environment context.

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `env:user_agent` | str | User agent string | `"ESDE/1.0"` |
| `env:accept_language` | str | Accept-Language header | `"en-US"` |
| `env:platform` | str | Platform identifier | `"linux"` |

---

### `legacy:` - Legacy Migration

Keys migrated from legacy `source_meta` format.

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `legacy:source_type` | str | Original source_type value | `"news"` |
| `legacy:language_profile` | str | Original language_profile value | `"en"` |

**Note:** The `legacy:` namespace is for migration only. New code should use appropriate semantic namespaces.

---

## Reserved Namespaces (Future)

The following namespaces are reserved for future use:

| Namespace | Intended Use |
|-----------|--------------|
| `audio:` | Audio file metadata |
| `video:` | Video file metadata |
| `image:` | Image file metadata |
| `pdf:` | PDF document metadata |
| `json:` | JSON structure metadata |
| `xml:` | XML structure metadata |
| `css:` | CSS statistics |
| `js:` | JavaScript statistics |

---

## Changelog

### v0.1.0 (2026-01-24)

- Initial namespace definitions
- Defined forbidden namespaces (INV-SUB-002)
- Added: `html:`, `text:`, `meta:`, `time:`, `struct:`, `env:`, `legacy:`

---

## Contributing

To add a new namespace or key:

1. Fork the repository
2. Add the namespace/key to this file
3. Update `traces.py` if new validation rules are needed
4. Create a PR with:
   - Clear description of what the namespace captures
   - Justification that values are machine-observable (INV-SUB-003)
   - Example values
5. Request review

**Reminder:** All trace values must be machine-observable facts.
Human interpretation or semantic classification is NOT allowed (INV-SUB-002).
