from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="MovieLens Dashboard (Week 3)", layout="wide")


@st.cache_data(ttl=3600)
def load_movie_ratings() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[2] / "data" / "movie_ratings.csv"
    df = pd.read_csv(data_path)
    return df


@st.cache_data(ttl=3600)
def load_movie_ratings_ec() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[2] / "data" / "movie_ratings.csv"
    df = pd.read_csv(data_path)
    return df


def render_header() -> None:
    st.title("MovieLens Dashboard")
    st.caption("Week 3 — EDA and Dashboards")


def render_sidebar(df: Optional[pd.DataFrame]) -> dict:
    with st.sidebar:
        st.header("Controls")
        show_raw = st.toggle("Show raw data preview", value=False)
        controls = {"show_raw": show_raw}
        return controls


def main() -> None:
    render_header()

    # Load data
    try:
        df = load_movie_ratings()
    except FileNotFoundError:
        st.error(
            "Could not find data file at ../data/movie_ratings.csv. "
            "Verify the project layout matches the course repo."
        )
        return

    controls = render_sidebar(df)

    if controls.get("show_raw"):
        with st.expander("Raw data (first 50 rows)", expanded=False):
            st.dataframe(df.head(50), use_container_width=True)

    tabs = st.tabs(
        [
            "Q1: Genre breakdown",
            "Q2: Avg rating by genre",
            "Q3: Avg rating by release year",
            "Q4: Top movies",
            "Q5: Ratings vs Age (EC)",
            "Q6: Ratings Volume vs Avg (EC)",
            "Q7: Cleaning genres (EC)",
        ]
    )

    # Q1
    with tabs[0]:
        st.subheader("Q1: What's the breakdown of genres for the movies that were rated?")
        st.caption("Pie chart of rating counts by pre-exploded 'genres'.")

        min_pct = st.slider("Group slices under this percentage into 'Other'", 0.0, 10.0, 2.0, 0.5)

        genre_counts = (
            df.groupby("genres", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        total = genre_counts["count"].sum()
        genre_counts["pct"] = 100 * genre_counts["count"] / max(total, 1)

        major = genre_counts[genre_counts["pct"] >= min_pct].copy()
        minor = genre_counts[genre_counts["pct"] < min_pct]
        if not minor.empty:
            other_row = pd.DataFrame({
                "genres": ["Other"],
                "count": [int(minor["count"].sum())],
                "pct": [minor["pct"].sum()],
            })
            display_df = pd.concat([major, other_row], ignore_index=True)
        else:
            display_df = major

        fig = px.pie(display_df, names="genres", values="count", title="Composition of Ratings by Genre")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    # Q2
    with tabs[1]:
        st.subheader("Q2: Which genres have the highest viewer satisfaction?")
        st.caption("Mean rating by genre.")

        min_count = st.number_input("Minimum ratings per genre", 0, 10000, 50, 10)
        sort_order = st.radio("Sort order", ["Descending", "Ascending"], horizontal=True)

        genre_stats = (
            df.groupby("genres", dropna=False)
            .agg(mean_rating=("rating", "mean"), n_ratings=("rating", "size"))
            .reset_index()
        )
        filtered = genre_stats[genre_stats["n_ratings"] >= min_count].copy()
        ascending = sort_order == "Ascending"
        filtered = filtered.sort_values("mean_rating", ascending=ascending)

        fig2 = px.bar(filtered, x="genres", y="mean_rating", hover_data={"n_ratings": True})
        fig2.update_layout(xaxis_title="Genre", yaxis_title="Average Rating")
        st.plotly_chart(fig2, use_container_width=True)

    # Q3
    with tabs[2]:
        st.subheader("Q3: How does mean rating change across movie release years?")
        st.caption("Line chart of mean rating by release year.")

        min_year, max_year = int(df["year"].min()), int(df["year"].max())
        year_range = st.slider("Year range", min_year, max_year, (min_year, max_year))
        min_count_year = st.number_input("Minimum ratings per year", 0, 100000, 50, 10)
        smooth_window = st.slider("Rolling mean window (years)", 1, 9, 1, 1)

        year_stats = (
            df.groupby("year", dropna=False)
            .agg(mean_rating=("rating", "mean"), n_ratings=("rating", "size"))
            .reset_index()
        )
        lo, hi = year_range
        mask = (year_stats["year"] >= lo) & (year_stats["year"] <= hi)
        year_filtered = year_stats[mask & (year_stats["n_ratings"] >= min_count_year)].copy()
        year_filtered = year_filtered.sort_values("year")

        if smooth_window > 1 and not year_filtered.empty:
            year_filtered["mean_rating_smoothed"] = (
                year_filtered["mean_rating"].rolling(window=smooth_window, center=True).mean()
            )
        else:
            year_filtered["mean_rating_smoothed"] = year_filtered["mean_rating"]

        fig3 = px.line(year_filtered, x="year", y="mean_rating_smoothed", hover_data={"n_ratings": True})
        st.plotly_chart(fig3, use_container_width=True)

    # Q4
    with tabs[3]:
        st.subheader("Q4: Top movies by average rating")
        st.caption("Compare top N movies at different rating thresholds.")

        min_ratings_movie = st.number_input("Minimum number of ratings per movie", 1, 100000, 50, 10)
        top_n = st.slider("Top N movies", 3, 25, 5, 1)

        movie_stats = (
            df.groupby(["movie_id", "title"], dropna=False)
            .agg(mean_rating=("rating", "mean"), n_ratings=("rating", "size"))
            .reset_index()
        )
        movie_filtered = movie_stats[movie_stats["n_ratings"] >= min_ratings_movie].copy()
        top_movies = movie_filtered.sort_values(["mean_rating", "n_ratings"], ascending=[False, False]).head(top_n)

        fig4 = px.bar(top_movies, y="title", x="mean_rating", orientation="h", hover_data={"n_ratings": True})
        st.plotly_chart(fig4, use_container_width=True)

    # Q5 EXTRA CREDIT
    with tabs[4]:
        st.subheader("Q5 (EC): Ratings vs Age for Selected Genres")
        st.caption("See how ratings change with viewer age across genres.")

        genres_to_compare = st.multiselect("Select up to 4 genres", options=df["genres"].unique(), default=df["genres"].unique()[:4])

        age_stats = (
            df[df["genres"].isin(genres_to_compare)]
            .groupby(["age", "genres"])
            .agg(mean_rating=("rating", "mean"), n_ratings=("rating", "size"))
            .reset_index()
        )

        fig5 = px.line(age_stats, x="age", y="mean_rating", color="genres", markers=True, hover_data={"n_ratings": True})
        st.plotly_chart(fig5, use_container_width=True)

    # Q6 EXTRA CREDIT
    with tabs[5]:
        st.subheader("Q6 (EC): Correlation between number of ratings and mean rating per genre")

        genre_stats = (
            df.groupby("genres")
            .agg(mean_rating=("rating", "mean"), n_ratings=("rating", "size"))
            .reset_index()
        )

        fig6 = px.scatter(genre_stats, x="n_ratings", y="mean_rating", text="genres", trendline="ols")
        st.plotly_chart(fig6, use_container_width=True)

    # Q7 EXTRA CREDIT
    with tabs[6]:
        st.subheader("Q7 (EC): Clean genres from raw dataset")

        try:
            df_ec = load_movie_ratings_ec()
            st.write("Preview of raw dataset:", df_ec.head())
            # Example cleaning: split genres by "|", explode rows
            df_ec["genres"] = df_ec["genres"].fillna("(no genres listed)")
            df_ec_cleaned = df_ec.assign(genres=df_ec["genres"].str.split("|")).explode("genres")
            st.write("Preview of cleaned dataset:", df_ec_cleaned.head())
        except FileNotFoundError:
            st.error("Could not find extra credit dataset movie_ratings_EC.csv")


if __name__ == "__main__":
    main()
