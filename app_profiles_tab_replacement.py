    # ────────────────────────────────────────
    # Tab 1: Profiles (W3)
    # ────────────────────────────────────────
    with tab_profiles:
        if w3v:
            profiles = w3v.get("profiles", {})
            dim_names = w3v.get("dim_names", [])

            st.markdown("### Condition Profiles (W3)")
            st.markdown(f"*{len(profiles)} conditions × {len(dim_names)} dimensions. "
                        "z-score shows how each condition differs from the global mean.*")

            # ── Filters ──
            # Extract article prefixes for filtering
            all_articles = sorted(set(
                cid.split("__")[0] for cid in profiles.keys() if "__" in cid
            ))

            filter_col1, filter_col2, filter_col3 = st.columns([3, 2, 2])
            with filter_col1:
                selected_articles = st.multiselect(
                    "Filter by article",
                    options=all_articles,
                    default=[],
                    placeholder="All articles",
                )
            with filter_col2:
                sort_by = st.selectbox(
                    "Sort by",
                    ["token_count (desc)", "condition name (asc)", "max |z| (desc)"],
                    index=0,
                )
            with filter_col3:
                page_size = st.selectbox(
                    "Show per page",
                    [20, 50, 100, "All"],
                    index=0,
                )

            # ── Apply filters ──
            filtered_profiles = {}
            for cid, prof in profiles.items():
                if selected_articles:
                    article_prefix = cid.split("__")[0] if "__" in cid else cid
                    if article_prefix not in selected_articles:
                        continue
                filtered_profiles[cid] = prof

            # ── Sort ──
            if sort_by.startswith("token_count"):
                sorted_items = sorted(
                    filtered_profiles.items(),
                    key=lambda x: x[1].get("token_count", 0),
                    reverse=True,
                )
            elif sort_by.startswith("condition"):
                sorted_items = sorted(filtered_profiles.items(), key=lambda x: x[0])
            else:  # max |z|
                def max_abs_z(item):
                    z_vec = item[1].get("z_score_vector", [])
                    return max((abs(z) for z in z_vec), default=0)
                sorted_items = sorted(
                    filtered_profiles.items(),
                    key=max_abs_z,
                    reverse=True,
                )

            total_filtered = len(sorted_items)
            st.caption(f"Showing {total_filtered} of {len(profiles)} conditions"
                       + (f" (filtered by: {', '.join(selected_articles)})"
                          if selected_articles else ""))

            # ── Pagination ──
            if page_size == "All":
                effective_page_size = total_filtered
            else:
                effective_page_size = int(page_size)

            total_pages = max(1, (total_filtered + effective_page_size - 1) // effective_page_size)

            if total_pages > 1:
                current_page = st.number_input(
                    f"Page (1–{total_pages})",
                    min_value=1, max_value=total_pages, value=1, step=1,
                )
            else:
                current_page = 1

            start_idx = (current_page - 1) * effective_page_size
            end_idx = min(start_idx + effective_page_size, total_filtered)
            page_items = sorted_items[start_idx:end_idx]

            # ── Heatmap (current page only) ──
            try:
                import pandas as pd
                import altair as alt

                heatmap_rows = []
                for cid, prof in page_items:
                    z_vec = prof.get("z_score_vector", [])
                    for i, val in enumerate(z_vec):
                        dim_name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
                        heatmap_rows.append({
                            "condition": cid,
                            "dimension": dim_name,
                            "z_score": round(val, 3),
                            "dim_idx": i,
                        })

                if heatmap_rows:
                    df_heat = pd.DataFrame(heatmap_rows)

                    n_rows = len(page_items)
                    heatmap = (
                        alt.Chart(df_heat)
                        .mark_rect()
                        .encode(
                            x=alt.X(
                                "dimension:N",
                                sort=alt.EncodingSortField(field="dim_idx"),
                                title="Feature Dimension",
                            ),
                            y=alt.Y(
                                "condition:N", title="Condition",
                                sort=[cid for cid, _ in page_items],
                            ),
                            color=alt.Color(
                                "z_score:Q",
                                scale=alt.Scale(scheme="redblue", domainMid=0),
                                legend=alt.Legend(title="z-score"),
                            ),
                            tooltip=["condition", "dimension", "z_score"],
                        )
                        .properties(height=max(150, n_rows * 22))
                    )

                    st.altair_chart(heatmap, use_container_width=True)
                    st.caption("🔴 Red = above average · 🔵 Blue = below average")
            except ImportError:
                st.warning("Install `altair` and `pandas` for heatmap.")

            # ── Top Features per Condition (current page only) ──
            st.markdown("---")
            st.markdown("#### Top Features per Condition")

            for cid, prof in page_items:
                token_count = prof.get("token_count", 0)
                top_pos = prof.get("top_positive", [])
                top_neg = prof.get("top_negative", [])

                with st.expander(f"**{cid}** ({token_count:,} tokens)"):
                    col_p, col_n = st.columns(2)
                    with col_p:
                        st.markdown("**↑ Overrepresented**")
                        for item in top_pos[:5]:
                            name = item.get("name", f"dim_{item.get('dim', '?')}")
                            z = item.get("z_score", 0)
                            st.markdown(f"- `{name}` z={z:+.3f}")
                    with col_n:
                        st.markdown("**↓ Underrepresented**")
                        for item in top_neg[:5]:
                            name = item.get("name", f"dim_{item.get('dim', '?')}")
                            z = item.get("z_score", 0)
                            st.markdown(f"- `{name}` z={z:+.3f}")

        elif conditions_token:
            st.markdown("### Resonating Tokens per Condition (W3)")
            st.markdown(f"*{len(conditions_token)} conditions. "
                        "S-score measures how strongly a token associates with a condition.*")

            if articles_token and isinstance(articles_token, dict):
                first_val = next(iter(articles_token.values()), {})
                if isinstance(first_val, dict) and "resonance_vector" in first_val:
                    st.markdown("#### Article Resonance Vectors")
                    try:
                        import pandas as pd
                        import altair as alt

                        res_rows = []
                        for aid, art_data in articles_token.items():
                            rv = art_data.get("resonance_vector", {})
                            for cond, val in rv.items():
                                res_rows.append({
                                    "article": aid,
                                    "condition": cond,
                                    "resonance": round(val, 4),
                                })

                        if res_rows:
                            df_res = pd.DataFrame(res_rows)
                            res_chart = (
                                alt.Chart(df_res)
                                .mark_rect()
                                .encode(
                                    x=alt.X("condition:N", title="Condition"),
                                    y=alt.Y("article:N", title="Article"),
                                    color=alt.Color(
                                        "resonance:Q",
                                        scale=alt.Scale(scheme="viridis"),
                                        legend=alt.Legend(title="Resonance"),
                                    ),
                                    tooltip=["article", "condition", "resonance"],
                                )
                                .properties(
                                    height=max(120, len(articles_token) * 30)
                                )
                            )
                            st.altair_chart(res_chart, use_container_width=True)
                    except ImportError:
                        pass
                    st.markdown("---")

            st.markdown("#### Top Tokens per Condition")
            for cid, cond_data in sorted(conditions_token.items()):
                positive = cond_data.get("positive", [])
                negative = cond_data.get("negative", [])
                with st.expander(f"**{cid}**"):
                    col_p, col_n = st.columns(2)
                    with col_p:
                        st.markdown("**↑ Positive (attracted)**")
                        for item in positive[:8]:
                            token = item.get("token", "?")
                            score = item.get("s_score", 0)
                            st.markdown(f"- `{token}` s={score:+.4f}")
                    with col_n:
                        st.markdown("**↓ Negative (repelled)**")
                        for item in negative[:8]:
                            token = item.get("token", "?")
                            score = item.get("s_score", 0)
                            st.markdown(f"- `{token}` s={score:+.4f}")
        else:
            st.info("No profile data available.")
