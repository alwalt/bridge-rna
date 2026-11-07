# NASA HDS style reference

This local reference separates NASA design tokens from the scientific category colors used in the animations.

## Fonts used in the videos

| Role | Font | Slide size |
|---|---|---|
| GTEx / TCGA heading | Inter Bold | 48 pt |
| UMAP axis labels and tissue/cancer legend | Inter Regular | 24 pt |
| Short dataset description | Public Sans Regular | 28 pt |

HDS also uses DM Mono for code/technical treatments; these videos do not use it.
The font files and their open-source licenses are installed in `~/.local/share/fonts/NASA-HDS/` on this machine. They are not duplicated in the benchmark. On other machines, install Inter and Public Sans before regenerating. Video text is rendered into frames, so viewers do not need the fonts installed.

## Colors

`tokens.json` records NASA primitive colors and the categorical dark palette for reference. **The videos use the notebook's original tab10 point colors, not NASA category colors.** The current videos use the notebook’s 10 organ-corresponding tissue/cancer pairs with one shared color per pair. Cortex/GBM is an approximate brain reference; organ correspondence does not imply matched patients or cell types.

The presentation retains its NASA HDS Carbon 90 `#17171b` background, white text, `#bfc6d2` subtitle, and `#b4bdcb` axes. The manuscript notebook figure retains its own typography and colors.

## Official sources

- [HDS typography](https://nasa.github.io/hds-core/?path=/docs/foundations-typography--docs)
- [Typography source](https://github.com/nasa/hds-core/blob/main/stories/foundations/Typography.mdx)
- [Color tokens](https://github.com/nasa/hds-core/blob/main/stories/helpers/hdsColors.js)
- [Data visualization colors](https://github.com/nasa/hds-core/blob/main/stories/helpers/hdsDatavizColors.js)

These are web design references adapted to a Hyperwall presentation, not a claim of NASA approval. This shared reference lives at `viz/nasa_style/` in the BridgeRNA repository. The `active_video_palette` in `tokens.json` records the BridgeRNA example and is not a NASA brand palette.
