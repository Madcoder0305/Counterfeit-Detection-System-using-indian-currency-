# CurrencyGuard ₹ — Design System

## Brand & Style
The design system is engineered to evoke absolute confidence and technical precision. Aimed at financial institutions and high-volume cash handlers, the visual language balances the authority of traditional banking with the cutting-edge intelligence of AI-powered detection. 

The aesthetic follows a **Premium Corporate** approach with **Glassmorphic** accents. It utilizes a sleek, dark charcoal background to reduce eye strain, while using a trustworthy Blue/Indigo as the primary brand color and vibrant semantic accents to signal status instantaneously.

## Colors
This design system utilizes a "Deep Charcoal & Trust Blue" palette.

- **Background (Charcoal):** `#0A0D14` for the core application canvas.
- **Surface Level 1:** `#111622` for the main containers, bordered by `#1C2333`.
- **Accent (Trust Blue):** `#3B82F6` for primary actions, active tabs, and highlights. Universally recognized as secure and professional.
- **Semantic Alerts:** 
  - **Genuine / Success:** `#10B981` (Emerald).
  - **Counterfeit / Alert:** `#EF4444` (Crimson).
  - **Warning / Low Confidence:** `#F59E0B` (Amber).
- **Text:** `#F1F5F9` (Primary), `#94A3B8` (Secondary/Dim).
- **Hover/Active Surfaces:** `#1E293B`

## Typography
The typography system prioritizes legibility and technical rigor.
- **Body & Headings:** `Inter`, system-ui, sans-serif
- **Data & Numbers:** `JetBrains Mono`, monospace (for currency values, confidence scores, transaction IDs)

## Layout & Spacing
- **Grid:** 12-column grid for the main stage. 
- **Margins:** 24px outer margins for desktop and 16px for mobile.
- **Rhythm:** Apply a 4px base unit. Component internals use 12px or 16px padding.

## Elevation & Depth
Depth is created through **Tonal Layering** rather than heavy shadows. 

- **Level 0 (Base):** Deep Charcoal background.
- **Level 1 (Card/Surface):** Lighter Charcoal/Navy for cards.
- **Outlines:** All elevated elements feature a 1px solid border (`#1C2333`).

## Shapes
- **Primary Radius:** `0.5rem` (8px) for standard buttons and inputs.
- **Container Radius:** `0.75rem` (12px) to `1rem` (16px) for cards, maps, viewfinders.

## Components

- **Buttons:** Primary buttons use `#3B82F6` background with White text. Secondary buttons use a `#1C2333` outline with no fill.
- **Status Chips:** 
  - "Genuine": Emerald Green tint background with solid Emerald text.
  - "Counterfeit": High-contrast Crimson.
- **Input Fields:** `#111622` background with a 1px `#1C2333` border. Focus transitions to Trust Blue (`#3B82F6`).
- **Data Tables:** Row hovering uses `#1E293B`. 
- **Progress Gauges/Bars:** Stroke dynamically changes from Crimson (<40%) to Amber (40-84%) to Emerald (85%+).
