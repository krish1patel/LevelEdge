---
name: LevelEdge
description: Calibrated close-above probabilities drawn as a forecast fan, in a quiet Apple-style instrument panel.
colors:
  accent: "#0071e3"
  accent-ink: "#ffffff"
  accent-wash: "rgba(0, 113, 227, 0.08)"
  fan-inner: "rgba(0, 113, 227, 0.24)"
  fan-outer: "rgba(0, 113, 227, 0.12)"
  neutral-series: "#8e8e93"
  price-line: "#1d1d1f"
  warn: "#9a6700"
  bad: "#c4321f"
  bg: "#f5f5f7"
  surface: "#ffffff"
  surface-2: "#fbfbfd"
  ink: "#1d1d1f"
  ink-2: "#4b4b50"
  ink-3: "#6e6e73"
  hairline: "rgba(0, 0, 0, 0.09)"
  hairline-strong: "rgba(0, 0, 0, 0.16)"
  grid: "rgba(0, 0, 0, 0.06)"
  seg-track: "#e8e8ed"
  seg-thumb: "#ffffff"
  dark-bg: "#000000"
  dark-surface: "#1c1c1e"
  dark-surface-2: "#232326"
  dark-ink: "#f5f5f7"
  dark-ink-2: "#c7c7cc"
  dark-ink-3: "#a1a1a6"
  dark-accent: "#2997ff"
  dark-accent-ink: "#000000"
  dark-fan-inner: "rgba(41, 151, 255, 0.32)"
  dark-fan-outer: "rgba(41, 151, 255, 0.16)"
  dark-warn: "#d29922"
  dark-bad: "#ff6a55"
  dark-seg-track: "#2c2c2e"
  dark-seg-thumb: "#636366"
typography:
  display:
    fontFamily: "-apple-system, BlinkMacSystemFont, \"SF Pro Text\", \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "3rem"
    fontWeight: 600
    lineHeight: 1.08
    letterSpacing: "-0.02em"
    fontFeature: "\"tnum\""
  headline:
    fontFamily: "-apple-system, BlinkMacSystemFont, \"SF Pro Text\", \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "2rem"
    fontWeight: 600
    lineHeight: 1.08
    letterSpacing: "-0.02em"
  title:
    fontFamily: "-apple-system, BlinkMacSystemFont, \"SF Pro Text\", \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "1.0625rem"
    fontWeight: 600
    lineHeight: 1.47
    letterSpacing: "-0.01em"
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, \"SF Pro Text\", \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "1.0625rem"
    fontWeight: 400
    lineHeight: 1.47
    letterSpacing: "-0.01em"
  label:
    fontFamily: "-apple-system, BlinkMacSystemFont, \"SF Pro Text\", \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 500
    lineHeight: 1.47
    letterSpacing: "-0.01em"
rounded:
  card: "18px"
  inner: "12px"
  focus: "8px"
  control: "980px"
spacing:
  s1: "4px"
  s2: "8px"
  s3: "16px"
  s4: "24px"
  s5: "48px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.accent-ink}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "0 24px"
    height: "44px"
  button-primary-hover:
    backgroundColor: "color-mix(in srgb, #0071e3 88%, #1d1d1f)"
    textColor: "{colors.accent-ink}"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.ink-2}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "0 24px"
    height: "44px"
  button-ghost-hover:
    backgroundColor: "{colors.accent-wash}"
    textColor: "{colors.ink}"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.ink-3}"
    typography: "{typography.label}"
    padding: "0 8px"
    height: "44px"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "0 16px"
    height: "44px"
  segmented-track:
    backgroundColor: "{colors.seg-track}"
    rounded: "{rounded.control}"
    height: "44px"
  segmented-option:
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "0 16px"
    height: "44px"
  segmented-option-selected:
    backgroundColor: "{colors.seg-thumb}"
    textColor: "{colors.ink}"
  card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.card}"
    padding: "{spacing.s4}"
  tab:
    textColor: "{colors.ink-3}"
    typography: "{typography.body}"
    padding: "8px 16px"
    height: "44px"
  tab-selected:
    textColor: "{colors.ink}"
  badge:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "4px 10px"
  tooltip:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.inner}"
    padding: "8px 12px"
  banner:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.card}"
    padding: "{spacing.s3}"
---

# Design System: LevelEdge

## Overview

**Creative North Star: "The Quiet Instrument"**

LevelEdge looks like a first-party platform utility: system type, neutral greys, one blue, white cards on a pale grey field. It stays quiet so the data carries the meaning. The forecast fan is the one expressive shape on the page, and it is drawn in the same blue as everything that says "this is LevelEdge's answer". Anything shown for comparison (baselines, older model versions) is grey, so a single glance separates the forecast from its benchmarks.

Density is medium, set for repeated mid-session checks. One large chart, a narrow side column of editable levels and trust evidence, and tabs for the deeper views. Numbers use tabular figures throughout so values can be scanned in columns and keep their position when they change on a re-run. Motion is short and eased: values restyle in place, rows flash once, data dims while it refetches, and nothing slides or bounces.

Light and dark themes are both first class. The system preference is followed unless the user pins one with the theme control, and every colour token has a matched dark value.

**Key Characteristics:**
- System font stack, weight 600 for headings, rem-based 48/32/17/13 scale.
- One accent (system blue). Its alpha steps are the fan bands.
- White 18px cards with a hairline border and a soft two-layer shadow on a #f5f5f7 field (pure black and #1c1c1e in dark mode).
- Pill controls at 44px height. The segmented control follows the platform pattern.
- Charts are hand-built SVG with emphasis encoding: accent means LevelEdge, grey means comparison, and there is no categorical palette.

## Colors

Neutral platform greys with one blue voice, plus two signal colours that only appear next to text that explains them.

### Primary
- **System Blue** (accent): the only accent colour. Used for the Forecast button, the brand mark, the median forecast line, probability bars, feature-importance bars, LevelEdge series in diagnostics and the track record, links, selection, and the focus ring. It is 4.7:1 on white. In dark mode it shifts to the lighter blue (dark-accent), and text on it switches to black (dark-accent-ink).
- **Blue Wash** (accent-wash): an 8% tint of the accent. Used for ghost and quiet button hover fills and the one-shot row flash, and (at 35% opacity) to shade the forecast region of the fan chart.
- **Fan Inner / Fan Outer** (fan-inner, fan-outer): accent alpha steps for the 50% and 80% forecast ranges. Dark mode raises both steps (dark-fan-inner, dark-fan-outer) so the bands stay visible on #1c1c1e.

### Neutral
- **Comparison Grey** (neutral-series): the stroke and fill for every comparison series (volatility baseline, the v1 model in the track record). Graphics only. At about 3.3:1 on white it passes the 3:1 non-text minimum but never carries text.
- **Price Ink** (price-line): the historical price line and the spot dot. Same value as ink, so price reads as fact and the forecast reads as blue.
- **Page Field** (bg), **Card White** (surface), **Recessed** (surface-2, inline code chips only).
- **Ink** (ink) for primary text, **Secondary Ink** (ink-2, 8.4:1) for supporting copy and legends, **Tertiary Ink** (ink-3, 5.0:1 on white and 4.6:1 on the field) for labels, captions, axis text and placeholders.
- **Hairline / Strong Hairline** (hairline, hairline-strong): card borders and row dividers, and the stronger value for input borders, axis lines, tooltips and banners.
- **Grid** (grid): chart gridlines, session boundaries, and empty tracks of bars and skeletons.
- **Segment Track / Thumb** (seg-track, seg-thumb): the segmented control only.

### Signal
- **Caution Amber** (warn): event risk and trust cautions. Used for the earnings banner border and icon, the earnings marker line on the fan (labelled "Earnings <day>"), and the icon on the trust panel's range-coverage caution. 4.9:1 on white.
- **Error Red** (bad): error banner border and icon, invalid input border, inline error text.

### Named Rules
**The One Voice Rule.** Blue means LevelEdge's own forecast and the primary action. Comparisons are always Comparison Grey. A chart never gets a categorical palette. If you need to show a third series, change its line style or label it directly; don't add a hue.

**The Fan Is the Accent Rule.** The fan bands are alpha steps of the accent (inner over outer) with a solid 2px accent median. Never draw a fan, range or confidence band in another hue or as a gradient.

**The Amber Speaks With Words Rule.** Caution Amber always comes with text that says what the risk is: a banner sentence or a labelled marker. It is never decoration and never the only signal.

**The Grey Floor Rule.** Tertiary Ink (ink-3) is the lightest grey allowed for text. The brief's #86868b (3.6:1 on white) is not in the system, and Comparison Grey is for strokes and fills only.

## Typography

**Display Font:** the system stack (-apple-system, BlinkMacSystemFont, "SF Pro Text", then Segoe UI, Roboto, Helvetica Neue, Arial)
**Body Font:** the same stack
**Label/Mono Font:** none. Numerals use tabular figures rather than a mono face.

**Character:** platform-native and unbranded. Hierarchy comes from size and weight 600, not from typeface contrast. Everything has a slight negative tracking (-0.01em, -0.02em at display size).

### Hierarchy
- **Display** (600, 3rem / 48px, 1.08): the ticker and spot price at the top of the dashboard. The ticker is in Tertiary Ink and the price is in Ink. Steps down to Headline on phones.
- **Headline** (600, 2rem / 32px, 1.08): the summary heading at 640px and below. Not used elsewhere.
- **Title** (600, 1.0625rem / 17px): card headings, banner leads, level values, P values, emphasised table rows, and subsection headings inside cards.
- **Body** (400, 1.0625rem / 17px, 1.47): copy, inputs, buttons, tabs, tables and key/value values.
- **Label** (500, 0.8125rem / 13px): field labels, fact terms, column headers, card subtitles, help and footnotes, chart axis and level labels, legends, tooltips and segmented options. Sentence case, never uppercase-tracked.

### Named Rules
**The Tabular Rule.** Every number on screen (prices, percentages, bar counts, deltas, axis ticks) uses tabular numerals, so columns align and values don't move sideways when they change.

**The Rem Rule.** All type sizes come from the four rem tokens so text follows the reader's font-size setting. Don't set text in px.

## Layout

Spacing follows a 4/8/16/24/48 rhythm (s1 to s5). Card padding is 24px, which drops to 16px on phones. Gaps between cards are 24px. The summary sits 24px above the tab row, and the tab row sits 24px above the panels.

The shell is centred at a maximum of 1440px with 24px side padding. The forecast view is a two-column grid: a fluid chart column and a fixed 380px side column holding Levels, then Trust. Below 1100px it becomes a single column. Diagnostics and Track record use an even two-up card grid that stacks below 900px.

The top bar is sticky. It uses a frosted field (bg at 86% with saturate 180% and blur 20px) over a hairline bottom border, and wraps its controls as the width shrinks. At 640px and below the top bar stops being sticky, the controls take the full width on their own row, the segmented control becomes a two-by-two grid, the Forecast button stretches, and summary facts form a two-column grid. The fan chart is 500px tall on desktop and 300px below a 560px container width.

**The Trading Time Rule.** Time axes for price run in trading bars, one slot per bar. Overnight and weekend gaps take no width, and each new session is marked with a solid Grid hairline and a day label. The fan's length is the real number of bars left to the resolving close.

## Elevation & Depth

Depth is a light hybrid: tonal layering (white cards on a grey field) does most of the work, with one soft ambient shadow so cards lift a little off the page. There are no hard or offset shadows. The frosted sticky top bar is the only translucent layer.

### Shadow Vocabulary
- **Card ambient** (`box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.05)`; dark `0 1px 2px rgba(0,0,0,0.5), 0 8px 24px rgba(0,0,0,0.35)`): cards and chart tooltips.
- **Segment thumb** (`box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 1px rgba(0,0,0,0.04)`; dark `0 1px 3px rgba(0,0,0,0.4)`): the selected segment only.
- **Focus ring** (`box-shadow: 0 0 0 2px var(--surface), 0 0 0 4px var(--accent)`): a 2px accent ring offset by 2px of surface, on every `:focus-visible`. It replaces the outline and follows the control's radius (8px by default, the full pill on inputs).

### Named Rules
**The Hairline Card Rule.** A card always has three things: a white surface, a 1px hairline border and the ambient shadow. Never use a heavier border or a coloured card fill to add emphasis.

## Shapes

Corners are soft and consistent. Containers (cards and banners) use 18px. Small floating or placeholder pieces (tooltips and skeletons) use 12px. Every interactive control (buttons, inputs, selects, segmented control, badges) is a full pill (980px). Icon-only remove buttons are circles. Probability and importance bars are thin rounded tracks (6px and 8px tall). Strokes on charts are 1px for structure and 2px for data, with round caps and joins.

**The Pill Control Rule.** If you can click or type into it, it is a pill at least 44px tall. Containers are never pills, and controls are never 18px rectangles.

## Components

### Buttons
- **Shape:** full pill (980px), 44px tall, 24px horizontal padding, with an optional 16px icon in front of the label.
- **Primary:** accent fill with accent-ink text. Used once per view, for Forecast. On hover the accent is mixed 12% toward ink.
- **Ghost:** transparent with a strong hairline border and Secondary Ink text. Hover adds the Blue Wash fill and Ink text. Used for Add, Backtest (a toggle with aria-pressed) and the 44px square theme button.
- **Quiet:** no border, Label size, Tertiary Ink. Used for row removal as a 44px circle.
- **States:** pressed scales to 0.98. Disabled drops to 45% opacity. While forecasting, Primary swaps its icon for a spinner and its label for "Forecasting".

### Segmented control (signature)
- Follows the platform pattern. A grey pill track holds pill options, and each option is a full 44px hit target. The visible thumb is a `::before` inset 2px with the thumb fill and thumb shadow, so the target stays 44px while the thumb reads as a raised platform-style chip.
- The selected option uses Ink at 600. Unselected options use Secondary Ink at 500 and turn Ink on hover. Selection is exposed with aria-pressed.
- On phones it becomes a 2x2 grid with 22px-radius options.

### Inputs / Fields
- **Style:** pill, 44px, white surface, strong hairline border, 16px padding, Body size. Labels sit above in Label/Tertiary Ink, indented 16px to line up with the text inside.
- **Hover:** the border darkens to Tertiary Ink. **Focus:** the border turns accent and the focus ring appears.
- **Error:** the border is Error Red, with an inline Label-size message in Error Red that is linked through aria-describedby.
- Selects use the same pill with a drawn chevron in Tertiary Ink.

### Cards / Containers
- **Corner Style:** 18px. **Background:** surface. **Border:** hairline. **Shadow:** card ambient. **Padding:** 24px (16px on phones).
- The header is a Title heading with an optional Label-size subtitle in Tertiary Ink, followed by 16px of space before the content.

### Navigation (tabs)
- A text tab row over a full-width hairline. Body size, weight 500, 44px minimum height. The selected tab uses Ink with a 2px Ink underline, and the others use Tertiary Ink. Arrow, Home and End keys move between tabs.

### Section navigation
- The top bar's first row holds the brand, three section links (Portfolio, Forecaster, Research) and the theme button. Links are 44px pills in Body weight 500 and Tertiary Ink; the current section uses Ink at 600 on the Segment Track fill (`aria-current="page"`), hover adds the Blue Wash.
- The forecaster's controls sit on their own full-width second row, shown only in the Forecaster section. Inside it the text tab row (Forecast, Diagnostics, Track record) is unchanged.

### Portfolio view
- Same two-column grid as the forecast view: a fluid main column (live paper account, drawdown chart, growth chart, performance table with a Since-2007 / Since-2020 segmented control, crash table) and the 380px side column (This week's targets, then How far to trust this).
- Charts follow the One Voice Rule: the LevelEdge book is the accent, 60/40 is a solid grey line, SPY is a dashed grey line. The primary series is drawn last so it sits on top.
- The targets table reuses the Levels-panel row: symbol in Title weight with the ETF name below in Label, the trend state as words ("On"/"Off") with the 12-month return below, and a 72px accent bar with the weight in Title weight. Bars are drawn to 50% of the account, and the help line says so.
- The not-connected live account is a teaching empty state (a numbered setup list), not a blank card.

### Research view
- One card per experiment, two columns (text left, numbers or chart right), stacking below 900px. Text sits in a 60ch column.
- Each card carries a verdict line in Title weight with a leading icon (check, cross, or minus). The verdict is always words; the icon is Tertiary Ink except Adopted, which uses the accent.
- The strategy-search scatter uses hollow grey dots for the other variants and one filled accent dot for the selected variant, unconnected.

### Levels panel
- A table of level rows. Each row shows the level in Title weight with its percent move below in Label/Tertiary Ink, a 72x6px probability bar with an accent fill, the P value in Title weight, and a delta since the last run (an up or down arrow plus the point change, with an aria-label in words).
- On a re-run, rows restyle in place without reflowing. Changed rows flash the Blue Wash once over 900ms, and bar widths ease over 250ms.

### Fan chart (signature)
- History in Price Ink at 2px flows into the fan at "Now", which is marked by a strong hairline. The forecast region is shaded with the Blue Wash. The 80% and 50% bands use Fan Outer and Fan Inner, and the median is a 2px accent line. The end label is the resolving bar close time.
- User levels are dashed lines (4 4, Tertiary Ink at 70%) that end in an Ink dot at the resolving bar. Each carries a right-hand label giving P in Title weight and the level in Secondary Ink, separated by leader lines so labels never collide. Levels outside the plotted range show a small triangle in place of the line.
- Gridlines, session boundaries and axes are solid hairlines. Earnings inside the window show as an amber 1.5px vertical line with a text label.
- Hovering shows a crosshair and a tooltip with the quantiles. The legend is written out, and the whole chart has a text summary as its accessible name.

### Comparison charts (XYChart)
- The LevelEdge series is accent and comparisons are Comparison Grey, as 2px lines or 4.5px dots with a surface outline. Reference lines (the spot line and the calibration diagonal) are strong hairlines. A legend appears when there is more than one series.

### Banners
- 18px radius, surface fill, 16px padding, Body size, a leading 18px icon and a bold lead sentence. The tone shows in the border (the signal colour at 45 to 50% mix) and the icon colour, and the text always states the issue. Errors use role alert and cautions use role status.

### Loading and refetch
- **First load:** skeletons (12px radius, a shimmering Grid-to-Hairline gradient) take the place of the summary, the chart (500px, or 300px on phones), the trust card, diagnostics and the track record.
- **Refetch:** the chart, level table and key/value lists dim to 50% over 200ms. Inputs, tabs and level editing stay live. The previous result stays on screen until the new one arrives.

### Motion
- All transitions use the shared ease (cubic-bezier(0.25, 0.1, 0.25, 1)), mostly at 200ms (120ms for press and tooltip). Under reduced motion, every animation and transition collapses to 1ms.

## Do's and Don'ts

### Do:
- **Do** draw LevelEdge's own series in the accent and every comparison in Comparison Grey. When two series need telling apart beyond colour, also use line style or direct labels.
- **Do** keep dashed strokes for reference lines that sit over the data: user price levels on the fan, and the volatility-baseline curve in diagnostics, where the dash is the cue that doesn't rely on colour. Gridlines, session boundaries and axes stay solid hairlines.
- **Do** pair every signal colour with words (a banner sentence or a labelled marker).
- **Do** make every control at least 44px tall, and use the shared focus ring on all of them.
- **Do** dim stale data to 50% during a refetch without locking any inputs, and use skeletons only for the first load.
- **Do** use tabular numerals for every number and rem tokens for every font size.

### Don't:
- **Don't** add a categorical palette, a second accent or a gradient to any chart.
- **Don't** use dashed gridlines or dashed session boundaries.
- **Don't** set text in Comparison Grey or in any grey lighter than Tertiary Ink.
- **Don't** use Caution Amber for anything except event risk and trust cautions, and never without explanatory text.
- **Don't** put a spinner or overlay over the chart during a refetch, or disable the level editor while one is running.
- **Don't** give cards heavy borders, coloured fills or hard offset shadows.
