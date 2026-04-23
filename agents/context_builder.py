"""
Builds dataset context to inject into prompts
"""

import pandas as pd


class DatasetContext:
    """Extracts useful metadata from a DataFrame for LLM prompts"""
    
    def __init__(self, df: pd.DataFrame, column_descriptions: dict = None):
        """
        Args:
            df: The dataframe to analyze
            column_descriptions: Optional dict like {'revenue': 'Total sales in USD'}
        """
        self.df = df
        self.descriptions = column_descriptions or {}
    
    def build(self) -> str:
        """Build complete context string for prompts"""
        parts = [
            self._basic_info(),
            self._column_details(),
            self._sample_data()
        ]
        return "\n\n".join(parts)
    
    def build_short(self) -> str:
        """Shorter version for prompts where space matters (like planner)"""
        parts = [
            self._basic_info(),
            self._column_details()
        ]
        return "\n\n".join(parts)
    
    def _basic_info(self) -> str:
        """Dataset overview"""
        return f"""DATASET OVERVIEW:
- Total rows: {len(self.df)}
- Total columns: {len(self.df.columns)}
- Columns: {', '.join(self.df.columns.tolist())}"""
    
    def _column_details(self) -> str:
        """Details about each column"""
        lines = ["COLUMN DETAILS:"]
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            desc = self.descriptions.get(col, "")
            
            # Numeric columns: show range
            if pd.api.types.is_numeric_dtype(self.df[col]):
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                mean_val = self.df[col].mean()
                stats = f"range: {min_val} to {max_val}, avg: {mean_val:.2f}"
            
            # Categorical/string columns: show unique values (if few)
            else:
                unique_vals = self.df[col].unique()
                if len(unique_vals) <= 10:
                    stats = f"values: {list(unique_vals)}"
                else:
                    stats = f"{len(unique_vals)} unique values, examples: {list(unique_vals[:3])}"
            
            # Build line
            line = f"- {col} ({dtype}): {stats}"
            if desc:
                line += f" | {desc}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _sample_data(self) -> str:
        """Sample rows"""
        sample = self.df.head(3).to_string(index=False)
        return f"SAMPLE DATA (first 3 rows):\n{sample}"