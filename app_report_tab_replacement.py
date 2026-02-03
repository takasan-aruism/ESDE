    # ────────────────────────────────────────
    # Tab 6: Report
    # ────────────────────────────────────────
    with tab_report:
        if report:
            with st.expander("📝 Markdown Report (report.md)", expanded=False):
                st.markdown(report)
        else:
            st.info("No report.md found.")

        # ── Structure Statistics (visual) ──
        if stats and stats.get("articles"):
            st.markdown("---")
            st.markdown("### 📐 Article Structure")
            st.markdown("*Sentence-level writing patterns. "
                        "No classification, no evaluation — numbers and shapes only.*")

            articles_data = stats["articles"]

            try:
                import pandas as pd
                import altair as alt

                # Build DataFrame from structure_stats.json
                rows = []
                for aid, s in sorted(articles_data.items()):
                    rows.append({
                        "article": aid,
                        "tokens": s.get("total_tokens", 0),
                        "sents": s.get("total_sentences", 0),
                        "paras": s.get("total_paragraphs", 0),
                        "avg": s.get("avg_sentence_length", 0),
                        # v1.1 fields — fallback to v1.0-safe defaults
                        "median": s.get("median_sentence_length",
                                        s.get("avg_sentence_length", 0)),
                        "std": s.get("std_sentence_length", 0),
                        "cv": s.get("cv_sentence_length", 0),
                        "iqr": s.get("iqr_sentence_length", 0),
                        "skew": s.get("skewness_sentence_length", 0),
                        "kurt": s.get("kurtosis_sentence_length", 0),
                    })
                df = pd.DataFrame(rows)

                # ── Corpus summary metrics ──
                n_articles = len(df)
                corpus_tokens = int(df["tokens"].sum())
                corpus_sents = int(df["sents"].sum())
                avg_cv = df["cv"].mean()

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                with mcol1:
                    st.metric("Articles", n_articles)
                with mcol2:
                    st.metric("Total Tokens", f"{corpus_tokens:,}")
                with mcol3:
                    st.metric("Total Sentences", f"{corpus_sents:,}")
                with mcol4:
                    st.metric("Mean CV", f"{avg_cv:.3f}" if avg_cv > 0 else "—")

                st.markdown("")

                # ── Table: GPT audit recommended layout ──
                # Tokens / Sents / Paras / Avg / Median / Std / CV
                has_v11 = df["cv"].sum() > 0

                display_cols = ["article", "tokens", "sents", "paras", "avg"]
                if has_v11:
                    display_cols.extend(["median", "std", "cv"])
                else:
                    display_cols.extend(["std"])

                df_display = df[display_cols].copy()
                for col in ["avg", "median", "std"]:
                    if col in df_display.columns:
                        df_display[col] = df_display[col].map(lambda x: f"{x:.1f}")
                if "cv" in df_display.columns:
                    df_display["cv"] = df_display["cv"].map(lambda x: f"{x:.3f}")

                st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, 35 * len(df_display) + 38),
                )

                # ── Chart 1: Avg vs CV scatter ──
                if has_v11:
                    st.markdown("#### Avg Length vs CV")
                    st.markdown("*Horizontal = sentence length, "
                                "Vertical = variability. Each dot is one article.*")

                    # Derive category from article_id prefix
                    df["category"] = df["article"].apply(
                        lambda x: x.split("_")[0] if "_" in x else "other"
                    )

                    scatter = (
                        alt.Chart(df)
                        .mark_circle(size=80, opacity=0.8)
                        .encode(
                            x=alt.X("avg:Q", title="Avg Sentence Length",
                                    scale=alt.Scale(zero=False)),
                            y=alt.Y("cv:Q", title="CV (std / avg)",
                                    scale=alt.Scale(zero=False)),
                            color=alt.Color(
                                "category:N", title="Category",
                                scale=alt.Scale(scheme="tableau10"),
                            ),
                            tooltip=[
                                "article", "avg", "median", "std",
                                "cv", "tokens", "sents",
                            ],
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(scatter, use_container_width=True)

                # ── Chart 2: Box plot of sentence lengths ──
                box_rows = []
                for aid, s in sorted(articles_data.items()):
                    lengths = s.get("sentence_lengths", [])
                    cat = aid.split("_")[0] if "_" in aid else "other"
                    for length in lengths:
                        box_rows.append({
                            "article": aid,
                            "category": cat,
                            "sentence_length": length,
                        })

                if box_rows:
                    st.markdown("#### Sentence Length Distribution")
                    st.markdown("*Box = IQR (Q25–Q75), "
                                "whiskers = min/max, line = median.*")

                    df_box = pd.DataFrame(box_rows)

                    boxplot = (
                        alt.Chart(df_box)
                        .mark_boxplot(extent="min-max", size=12)
                        .encode(
                            x=alt.X(
                                "article:N", title="Article",
                                sort=alt.EncodingSortField(field="article"),
                                axis=alt.Axis(labelAngle=-45, labelLimit=200),
                            ),
                            y=alt.Y(
                                "sentence_length:Q",
                                title="Sentence Length (tokens)",
                                scale=alt.Scale(zero=False),
                            ),
                            color=alt.Color(
                                "category:N", title="Category",
                                scale=alt.Scale(scheme="tableau10"),
                                legend=None,
                            ),
                        )
                        .properties(height=350)
                    )
                    st.altair_chart(boxplot, use_container_width=True)

                # ── Advanced metrics (expandable) ──
                if has_v11 and df["iqr"].sum() > 0:
                    with st.expander("Advanced metrics (IQR / Skewness / Kurtosis)"):
                        df_adv = df[["article", "iqr", "skew", "kurt"]].copy()
                        df_adv["iqr"] = df_adv["iqr"].map(lambda x: f"{x:.1f}")
                        df_adv["skew"] = df_adv["skew"].map(lambda x: f"{x:+.2f}")
                        df_adv["kurt"] = df_adv["kurt"].map(lambda x: f"{x:+.2f}")
                        st.dataframe(
                            df_adv,
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.caption(
                            "**Skewness:** positive = short-sentence heavy "
                            "+ occasional long. "
                            "**Kurtosis:** positive = peaked rhythm, "
                            "negative = diverse rhythm."
                        )

            except ImportError:
                st.warning(
                    "Install `pandas` and `altair` for visual display. "
                    "(`pip install pandas altair`)"
                )
                st.json(stats)

            # Raw data fallback
            with st.expander("Raw structure_stats.json"):
                st.json(stats)

        st.markdown("---")
        with st.expander("Raw analysis.json"):
            st.json(analysis)
