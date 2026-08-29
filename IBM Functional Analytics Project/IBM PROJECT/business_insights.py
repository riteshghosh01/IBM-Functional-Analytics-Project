from typing import Dict, List, Optional

import pandas as pd


def generate_business_insights(
    df: pd.DataFrame,
    revenue_col: Optional[str] = None,
    category_col: Optional[str] = None,
    region_col: Optional[str] = None,
    date_col: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Generate plain-language insights from the uploaded dataset."""
    insights = []

    if df.empty:
        return insights

    if revenue_col and category_col and revenue_col in df.columns and category_col in df.columns:
        category_summary = df.groupby(category_col, dropna=False)[revenue_col].sum().sort_values(ascending=False)
        if not category_summary.empty:
            top_category = category_summary.index[0]
            top_value = category_summary.iloc[0]
            insights.append({
                "type": "insight",
                "text": f"**Insight:** {top_category} generated the highest {revenue_col.lower()} with a total of {top_value:,.2f}.",
                "recommendation": f"**Recommendation:** Consider increasing inventory and marketing investment for {top_category}.",
            })

            low_category = category_summary.index[-1]
            low_value = category_summary.iloc[-1]
            insights.append({
                "type": "insight",
                "text": f"**Insight:** {low_category} generated the lowest {revenue_col.lower()} with a total of {low_value:,.2f}.",
                "recommendation": f"**Recommendation:** Review pricing, promotions, or demand issues for {low_category}.",
            })

    if revenue_col and region_col and revenue_col in df.columns and region_col in df.columns:
        region_summary = df.groupby(region_col, dropna=False)[revenue_col].sum().sort_values(ascending=False)
        if not region_summary.empty:
            top_region = region_summary.index[0]
            insights.append({
                "type": "insight",
                "text": f"**Insight:** {top_region} is the top performing region by {revenue_col.lower()}.",
                "recommendation": f"**Recommendation:** Increase focus on regional campaigns and supply planning for {top_region}.",
            })

    if revenue_col and date_col and revenue_col in df.columns and date_col in df.columns:
        try:
            temp = df[[date_col, revenue_col]].copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna(subset=[date_col, revenue_col])
            if not temp.empty:
                trend = temp.groupby(pd.Grouper(key=date_col, freq="M"))[revenue_col].sum().reset_index()
                if len(trend) > 1:
                    diff = trend[revenue_col].pct_change().fillna(0)
                    if diff.iloc[-1] > 0:
                        insights.append({
                            "type": "insight",
                            "text": "**Insight:** The latest monthly trend is increasing.",
                            "recommendation": "**Recommendation:** Maintain or increase promotion spend to capitalize on the positive growth trend.",
                        })
                    else:
                        insights.append({
                            "type": "insight",
                            "text": "**Insight:** The latest monthly trend is decreasing.",
                            "recommendation": "**Recommendation:** Review recent demand changes, pricing, or channel performance to identify the decline.",
                        })
        except Exception:
            pass

    if revenue_col and revenue_col in df.columns:
        numeric = pd.to_numeric(df[revenue_col], errors="coerce")
        if not numeric.empty:
            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1
            outliers = df[(numeric < (q1 - 1.5 * iqr)) | (numeric > (q3 + 1.5 * iqr))]
            if not outliers.empty:
                insights.append({
                    "type": "insight",
                    "text": "**Insight:** Unusual values were detected in the selected revenue/sales field.",
                    "recommendation": "**Recommendation:** Investigate outliers for data quality issues or extraordinary business events.",
                })

    return insights
