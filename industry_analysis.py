"""
Industry Rotation and Economic Cycle Analysis Tool
===================================================

This tool analyzes economic cycles and industry performance patterns to provide
sector rotation insights based on macroeconomic conditions.

SETUP INSTRUCTIONS:
------------------
1. Install required libraries:
   pip install pandas numpy matplotlib seaborn plotly scipy openpyxl xlrd

2. Save this script as: industry_analysis.py

3. Run: python industry_analysis.py

Note: xlrd library is needed to read .xls files (older Excel format)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set styling for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION - GOOGLE COLAB FILE PATH
# ============================================================================

# Your Excel file path in Google Colab
EXCEL_FILE = '/content/industry_analysis.xls'
SHEET_NAME = 'Sheet1'  # UPDATE THIS if your sheet has a different name

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

class EconomicDataLoader:
    """Loads and preprocesses economic and industry data"""
    
    def __init__(self, filepath, sheet_name='Sheet1'):
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.df = None
        self.macro_indicators = None
        self.industry_indicators = None
        
    def load_data(self):
        """Load data from Excel file"""
        print("Loading data from Excel...")
        self.df = pd.read_excel(self.filepath, sheet_name=self.sheet_name)
        
        # Clean column names - remove extra spaces
        self.df.columns = self.df.columns.str.strip()
        
        print(f"Loaded {len(self.df)} years of data")
        print(f"Date range: {self.df['Year'].min()} to {self.df['Year'].max()}")
        return self
    
    def identify_indicators(self):
        """Separate macro indicators from industry indicators"""
        
        # Macroeconomic indicators
        macro_cols = [
            'YoY % Change GDP at market prices',
            'YoY % Change GVA at basic prices',
            'YoY % Change CPI IW',
            'YoY % Change Credit to Private Sector',
            'YoY % Change in IIP'
        ]
        
        # All industry revenue columns
        industry_cols = [col for col in self.df.columns if 'Sector Revenue' in col]
        
        self.macro_indicators = ['Year'] + [col for col in macro_cols if col in self.df.columns]
        self.industry_indicators = ['Year'] + industry_cols
        
        print(f"\nMacro indicators: {len(self.macro_indicators)-1}")
        print(f"Industry sectors: {len(industry_cols)}")
        
        return self
    
    def get_clean_data(self):
        """Return cleaned dataframes"""
        macro_df = self.df[self.macro_indicators].copy()
        industry_df = self.df[self.industry_indicators].copy()
        
        # Handle missing values with forward fill then backward fill
        macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')
        industry_df = industry_df.fillna(method='ffill').fillna(method='bfill')
        
        return macro_df, industry_df


# ============================================================================
# 2. ECONOMIC CYCLE ANALYZER
# ============================================================================

class CycleAnalyzer:
    """Analyzes economic cycles and identifies phases"""
    
    def __init__(self, macro_df):
        self.macro_df = macro_df
        self.cycle_phases = None
        
    def identify_cycle_phases(self):
        """
        Identify economic cycle phases based on multiple indicators
        
        Phases:
        - Expansion: High growth, rising inflation, strong credit
        - Late Expansion: Growth slowing, inflation high, tight credit
        - Contraction: Negative/low growth, falling inflation
        - Recovery: Growth accelerating from low base
        """
        
        df = self.macro_df.copy()
        
        # Get GDP growth
        gdp_col = 'YoY % Change GDP at market prices'
        gdp_growth = df[gdp_col]
        
        # Calculate GDP growth percentiles for classification
        gdp_median = gdp_growth.median()
        gdp_75th = gdp_growth.quantile(0.75)
        gdp_25th = gdp_growth.quantile(0.25)
        
        # Calculate momentum (is growth accelerating or decelerating?)
        gdp_momentum = gdp_growth.diff()
        
        # Initialize phase column
        df['Cycle_Phase'] = ''
        
        for idx in df.index:
            growth = gdp_growth.iloc[idx]
            momentum = gdp_momentum.iloc[idx] if idx > 0 else 0
            
            # Classification logic
            if growth >= gdp_75th:
                if momentum > 0:
                    df.loc[idx, 'Cycle_Phase'] = 'Strong Expansion'
                else:
                    df.loc[idx, 'Cycle_Phase'] = 'Late Expansion'
            elif growth >= gdp_median:
                if momentum > 0:
                    df.loc[idx, 'Cycle_Phase'] = 'Early Expansion'
                else:
                    df.loc[idx, 'Cycle_Phase'] = 'Moderate Growth'
            elif growth >= gdp_25th:
                if momentum < 0:
                    df.loc[idx, 'Cycle_Phase'] = 'Early Slowdown'
                else:
                    df.loc[idx, 'Cycle_Phase'] = 'Recovery'
            else:
                if momentum < 0:
                    df.loc[idx, 'Cycle_Phase'] = 'Contraction'
                else:
                    df.loc[idx, 'Cycle_Phase'] = 'Early Recovery'
        
        self.cycle_phases = df[['Year', 'Cycle_Phase', gdp_col]]
        
        print("\nCycle Phase Distribution:")
        print(df['Cycle_Phase'].value_counts())
        
        return df
    
    def create_cycle_timeline(self):
        """Create visualization of cycle phases over time"""
        
        df = self.identify_cycle_phases()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: GDP Growth with phases highlighted
        gdp_col = 'YoY % Change GDP at market prices'
        
        # Color map for phases
        phase_colors = {
            'Strong Expansion': 'darkgreen',
            'Early Expansion': 'lightgreen',
            'Moderate Growth': 'yellow',
            'Late Expansion': 'orange',
            'Early Slowdown': 'lightsalmon',
            'Contraction': 'red',
            'Recovery': 'lightblue',
            'Early Recovery': 'skyblue'
        }
        
        ax1.plot(df['Year'], df[gdp_col], 'k-', linewidth=2, label='GDP Growth')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Color background by phase
        for i in range(len(df)-1):
            phase = df.iloc[i]['Cycle_Phase']
            ax1.axvspan(df.iloc[i]['Year'], df.iloc[i+1]['Year'], 
                       alpha=0.3, color=phase_colors.get(phase, 'gray'))
        
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('GDP Growth (%)', fontsize=12)
        ax1.set_title('Economic Cycle Phases (Based on GDP Growth & Momentum)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Multiple indicators
        if 'YoY % Change CPI IW' in df.columns:
            ax2_twin = ax2.twinx()
            ax2.plot(df['Year'], df[gdp_col], 'g-', linewidth=2, label='GDP Growth')
            ax2_twin.plot(df['Year'], df['YoY % Change CPI IW'], 'r-', 
                         linewidth=2, label='Inflation')
            
            ax2.set_xlabel('Year', fontsize=12)
            ax2.set_ylabel('GDP Growth (%)', color='g', fontsize=12)
            ax2_twin.set_ylabel('Inflation (%)', color='r', fontsize=12)
            ax2.set_title('GDP Growth vs Inflation', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.tight_layout()
        plt.savefig('economic_cycle_timeline.png', dpi=300, bbox_inches='tight')
        print("\nSaved: economic_cycle_timeline.png")
        plt.show()
        
        return df


# ============================================================================
# 3. INDUSTRY PERFORMANCE ANALYZER
# ============================================================================

class IndustryAnalyzer:
    """Analyzes industry performance across economic cycles"""
    
    def __init__(self, macro_df, industry_df):
        self.macro_df = macro_df
        self.industry_df = industry_df
        self.merged_df = None
        
    def merge_data(self):
        """Merge macro and industry data"""
        self.merged_df = pd.merge(self.macro_df, self.industry_df, on='Year')
        return self.merged_df
    
    def analyze_sector_sensitivity(self, gdp_col='YoY % Change GDP at market prices'):
        """
        Calculate how sensitive each industry is to GDP growth
        Higher correlation = more cyclical
        Lower correlation = more defensive
        """
        
        df = self.merge_data()
        
        # Get industry columns
        industry_cols = [col for col in df.columns if 'Sector Revenue' in col]
        
        # Calculate correlation with GDP
        correlations = {}
        for col in industry_cols:
            # Clean sector name
            sector_name = col.replace('YoY % Change in ', '').replace(' Sector Revenue', '')
            
            # Calculate correlation (handling NaN)
            valid_data = df[[gdp_col, col]].dropna()
            if len(valid_data) > 3:
                corr = valid_data[gdp_col].corr(valid_data[col])
                correlations[sector_name] = corr
        
        # Sort by correlation
        corr_df = pd.DataFrame.from_dict(correlations, orient='index', 
                                         columns=['GDP_Correlation'])
        corr_df = corr_df.sort_values('GDP_Correlation', ascending=False)
        
        # Classify sectors
        corr_df['Classification'] = corr_df['GDP_Correlation'].apply(
            lambda x: 'Highly Cyclical' if x > 0.6 
            else ('Moderately Cyclical' if x > 0.3 
            else ('Low Cyclicality' if x > 0 
            else 'Counter-Cyclical'))
        )
        
        print("\n" + "="*70)
        print("SECTOR CYCLICALITY ANALYSIS")
        print("="*70)
        print(corr_df)
        
        # Visualize
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = corr_df['GDP_Correlation'].apply(
            lambda x: 'darkgreen' if x > 0.6 else ('green' if x > 0.3 else 'orange')
        )
        
        bars = ax.barh(range(len(corr_df)), corr_df['GDP_Correlation'], color=colors)
        ax.set_yticks(range(len(corr_df)))
        ax.set_yticklabels(corr_df.index)
        ax.set_xlabel('Correlation with GDP Growth', fontsize=12)
        ax.set_title('Industry Cyclicality: Correlation with GDP Growth\n' +
                    '(Higher = More Cyclical, Lower = More Defensive)', 
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('sector_cyclicality.png', dpi=300, bbox_inches='tight')
        print("\nSaved: sector_cyclicality.png")
        plt.show()
        
        return corr_df
    
    def analyze_performance_by_cycle(self, cycle_df):
        """Analyze how industries perform in different cycle phases"""
        
        # Merge cycle phases with industry data
        df = pd.merge(cycle_df[['Year', 'Cycle_Phase']], 
                     self.industry_df, on='Year')
        
        # Get industry columns
        industry_cols = [col for col in df.columns if 'Sector Revenue' in col]
        
        # Calculate average performance by cycle phase
        results = {}
        
        for col in industry_cols:
            sector_name = col.replace('YoY % Change in ', '').replace(' Sector Revenue', '')
            phase_performance = df.groupby('Cycle_Phase')[col].mean()
            results[sector_name] = phase_performance
        
        # Create DataFrame
        performance_df = pd.DataFrame(results).T
        
        # Sort columns by typical cycle progression
        cycle_order = ['Contraction', 'Early Recovery', 'Recovery', 
                      'Early Expansion', 'Strong Expansion', 'Late Expansion', 
                      'Moderate Growth', 'Early Slowdown']
        
        # Keep only columns that exist
        available_phases = [phase for phase in cycle_order if phase in performance_df.columns]
        performance_df = performance_df[available_phases]
        
        print("\n" + "="*70)
        print("AVERAGE SECTOR PERFORMANCE BY CYCLE PHASE (%)")
        print("="*70)
        print(performance_df.round(2))
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(performance_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Avg Growth Rate (%)'})
        plt.title('Industry Performance Across Economic Cycle Phases\n' +
                 '(Average YoY Revenue Growth %)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Cycle Phase', fontsize=12)
        plt.ylabel('Industry Sector', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('performance_by_cycle_phase.png', dpi=300, bbox_inches='tight')
        print("\nSaved: performance_by_cycle_phase.png")
        plt.show()
        
        return performance_df
    
    def create_sector_comparison_chart(self):
        """Create interactive time series comparison of all sectors"""
        
        df = self.industry_df.copy()
        industry_cols = [col for col in df.columns if 'Sector Revenue' in col]
        
        fig = go.Figure()
        
        for col in industry_cols:
            sector_name = col.replace('YoY % Change in ', '').replace(' Sector Revenue', '')
            fig.add_trace(go.Scatter(
                x=df['Year'],
                y=df[col],
                name=sector_name,
                mode='lines',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Industry Revenue Growth Trends Over Time',
            xaxis_title='Year',
            yaxis_title='YoY Revenue Growth (%)',
            hovermode='x unified',
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        fig.write_html('sector_trends_interactive.html')
        print("\nSaved: sector_trends_interactive.html")
        print("(Open this file in your web browser for interactive exploration)")
        
        return fig
    
    def analyze_sectoral_correlations(self):
        """
        Analyze correlations between different sectors
        Helps identify diversification opportunities and sector clustering
        """
        
        df = self.industry_df.copy()
        industry_cols = [col for col in df.columns if 'Sector Revenue' in col]
        
        # Create clean sector names
        sector_data = {}
        for col in industry_cols:
            sector_name = col.replace('YoY % Change in ', '').replace(' Sector Revenue', '')
            sector_data[sector_name] = df[col]
        
        sector_df = pd.DataFrame(sector_data)
        
        # Calculate correlation matrix
        corr_matrix = sector_df.corr()
        
        print("\n" + "="*70)
        print("SECTORAL CORRELATION ANALYSIS")
        print("="*70)
        print("\nHighly Correlated Sector Pairs (>0.7):")
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], 
                                           corr_matrix.columns[j], 
                                           corr_val))
        
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        for sec1, sec2, corr in high_corr_pairs[:10]:
            print(f"  {sec1} <-> {sec2}: {corr:.3f}")
        
        print("\nLow/Negative Correlations (<0.3) - Diversification Opportunities:")
        low_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val < 0.3:
                    low_corr_pairs.append((corr_matrix.columns[i], 
                                          corr_matrix.columns[j], 
                                          corr_val))
        
        low_corr_pairs.sort(key=lambda x: x[2])
        for sec1, sec2, corr in low_corr_pairs[:10]:
            print(f"  {sec1} <-> {sec2}: {corr:.3f}")
        
        # Create correlation heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Sectoral Correlation Matrix\n(Higher values = sectors move together)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('sectoral_correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("\nSaved: sectoral_correlation_matrix.png")
        plt.show()
        
        return corr_matrix
    
    def analyze_multi_factor_sensitivity(self):
        """
        Analyze how sectors respond to multiple macro factors
        Not just GDP, but also inflation, credit growth, industrial production
        """
        
        df = self.merge_data()
        industry_cols = [col for col in df.columns if 'Sector Revenue' in col]
        
        # Define macro factors
        macro_factors = {
            'GDP Growth': 'YoY % Change GDP at market prices',
            'Inflation': 'YoY % Change CPI IW',
            'Credit Growth': 'YoY % Change Credit to Private Sector',
            'Industrial Production': 'YoY % Change in IIP'
        }
        
        # Calculate sensitivities
        sensitivity_data = []
        
        for ind_col in industry_cols:
            sector_name = ind_col.replace('YoY % Change in ', '').replace(' Sector Revenue', '')
            sector_sensitivities = {'Sector': sector_name}
            
            for factor_name, factor_col in macro_factors.items():
                if factor_col in df.columns:
                    valid_data = df[[factor_col, ind_col]].dropna()
                    if len(valid_data) > 3:
                        corr = valid_data[factor_col].corr(valid_data[ind_col])
                        sector_sensitivities[factor_name] = corr
                    else:
                        sector_sensitivities[factor_name] = np.nan
                else:
                    sector_sensitivities[factor_name] = np.nan
            
            sensitivity_data.append(sector_sensitivities)
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        sensitivity_df = sensitivity_df.set_index('Sector')
        
        print("\n" + "="*70)
        print("MULTI-FACTOR SENSITIVITY ANALYSIS")
        print("="*70)
        print("\nSector Sensitivity to Different Macro Factors:")
        print(sensitivity_df.round(3))
        
        # Create heatmap
        plt.figure(figsize=(10, 12))
        sns.heatmap(sensitivity_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Multi-Factor Sensitivity Analysis\n' + 
                 'How Different Sectors Respond to Macro Variables', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Macro Economic Factor', fontsize=12)
        plt.ylabel('Industry Sector', fontsize=12)
        plt.tight_layout()
        plt.savefig('multifactor_sensitivity.png', dpi=300, bbox_inches='tight')
        print("\nSaved: multifactor_sensitivity.png")
        plt.show()
        
        return sensitivity_df
    
    def analyze_sector_volatility(self):
        """
        Calculate and compare sector volatility (standard deviation of returns)
        High volatility = higher risk, Low volatility = more stable/defensive
        """
        
        df = self.industry_df.copy()
        industry_cols = [col for col in df.columns if 'Sector Revenue' in col]
        
        volatility_data = {}
        
        for col in industry_cols:
            sector_name = col.replace('YoY % Change in ', '').replace(' Sector Revenue', '')
            volatility_data[sector_name] = {
                'Volatility (Std Dev)': df[col].std(),
                'Mean Growth': df[col].mean(),
                'Min Growth': df[col].min(),
                'Max Growth': df[col].max(),
                'Coefficient of Variation': df[col].std() / abs(df[col].mean()) if df[col].mean() != 0 else np.nan
            }
        
        volatility_df = pd.DataFrame(volatility_data).T
        volatility_df = volatility_df.sort_values('Volatility (Std Dev)', ascending=False)
        
        print("\n" + "="*70)
        print("SECTOR VOLATILITY ANALYSIS")
        print("="*70)
        print("\nSector Risk Profile (sorted by volatility):")
        print(volatility_df.round(2))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Chart 1: Volatility ranking
        colors = ['red' if x > volatility_df['Volatility (Std Dev)'].median() 
                 else 'green' for x in volatility_df['Volatility (Std Dev)']]
        
        ax1.barh(range(len(volatility_df)), volatility_df['Volatility (Std Dev)'], 
                color=colors, alpha=0.7)
        ax1.set_yticks(range(len(volatility_df)))
        ax1.set_yticklabels(volatility_df.index)
        ax1.set_xlabel('Standard Deviation of Growth Rate (%)', fontsize=11)
        ax1.set_title('Sector Volatility (Risk)\nRed = High Risk, Green = Low Risk', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Chart 2: Risk-Return scatter
        ax2.scatter(volatility_df['Volatility (Std Dev)'], 
                   volatility_df['Mean Growth'],
                   s=150, alpha=0.6, c='blue')
        
        for sector in volatility_df.index:
            ax2.annotate(sector, 
                        (volatility_df.loc[sector, 'Volatility (Std Dev)'],
                         volatility_df.loc[sector, 'Mean Growth']),
                        fontsize=8, alpha=0.7)
        
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Volatility (Risk)', fontsize=11)
        ax2.set_ylabel('Average Growth (Return)', fontsize=11)
        ax2.set_title('Risk-Return Profile by Sector', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sector_volatility_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSaved: sector_volatility_analysis.png")
        plt.show()
        
        return volatility_df
    
    def analyze_sector_momentum(self):
        """
        Analyze recent momentum - which sectors are accelerating or decelerating
        """
        
        df = self.industry_df.copy()
        industry_cols = [col for col in df.columns if 'Sector Revenue' in col]
        
        # Calculate 3-year rolling averages for recent vs historical
        recent_window = 3
        historical_window = 10
        
        momentum_data = {}
        
        for col in industry_cols:
            sector_name = col.replace('YoY % Change in ', '').replace(' Sector Revenue', '')
            
            if len(df[col].dropna()) >= recent_window + historical_window:
                recent_avg = df[col].tail(recent_window).mean()
                historical_avg = df[col].head(-recent_window).mean()
                momentum = recent_avg - historical_avg
                
                momentum_data[sector_name] = {
                    'Recent Performance (3Y)': recent_avg,
                    'Historical Average': historical_avg,
                    'Momentum': momentum,
                    'Momentum Category': 'Accelerating' if momentum > 2 else (
                        'Stable' if momentum > -2 else 'Decelerating')
                }
        
        momentum_df = pd.DataFrame(momentum_data).T
        momentum_df = momentum_df.sort_values('Momentum', ascending=False)
        
        print("\n" + "="*70)
        print("SECTOR MOMENTUM ANALYSIS")
        print("="*70)
        print("\nMomentum = Recent Performance (3Y) - Historical Average")
        print("Positive = Sector is accelerating, Negative = Decelerating\n")
        print(momentum_df.round(2))
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['darkgreen' if x > 2 else ('green' if x > 0 else ('orange' if x > -2 else 'red')) 
                 for x in momentum_df['Momentum']]
        
        ax.barh(range(len(momentum_df)), momentum_df['Momentum'], color=colors, alpha=0.7)
        ax.set_yticks(range(len(momentum_df)))
        ax.set_yticklabels(momentum_df.index)
        ax.set_xlabel('Momentum (Recent vs Historical Performance)', fontsize=11)
        ax.set_title('Sector Momentum Analysis\n' +
                    'Green = Accelerating, Red = Decelerating', 
                    fontsize=13, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('sector_momentum.png', dpi=300, bbox_inches='tight')
        print("\nSaved: sector_momentum.png")
        plt.show()
        
        return momentum_df


# ============================================================================
# 4. ROTATION RECOMMENDER
# ============================================================================

class RotationRecommender:
    """Provides sector rotation recommendations based on current conditions"""
    
    def __init__(self, macro_df, industry_df, cycle_df, performance_df, sensitivity_df):
        self.macro_df = macro_df
        self.industry_df = industry_df
        self.cycle_df = cycle_df
        self.performance_df = performance_df
        self.sensitivity_df = sensitivity_df
    
    def get_current_recommendations(self, lookback_years=3):
        """
        Generate rotation recommendations based on recent economic trends
        """
        
        # Get recent data
        recent_macro = self.macro_df.tail(lookback_years)
        recent_industry = self.industry_df.tail(lookback_years)
        current_phase = self.cycle_df['Cycle_Phase'].iloc[-1]
        current_year = self.cycle_df['Year'].iloc[-1]
        
        print("\n" + "="*70)
        print("CURRENT ECONOMIC ASSESSMENT")
        print("="*70)
        print(f"Latest Year: {current_year}")
        print(f"Current Cycle Phase: {current_phase}")
        print(f"\nRecent Economic Indicators (Last {lookback_years} years avg):")
        
        # Calculate recent averages
        gdp_col = 'YoY % Change GDP at market prices'
        if gdp_col in recent_macro.columns:
            print(f"  GDP Growth: {recent_macro[gdp_col].mean():.2f}%")
        
        if 'YoY % Change CPI IW' in recent_macro.columns:
            print(f"  Inflation: {recent_macro['YoY % Change CPI IW'].mean():.2f}%")
        
        if 'YoY % Change Credit to Private Sector' in recent_macro.columns:
            print(f"  Credit Growth: {recent_macro['YoY % Change Credit to Private Sector'].mean():.2f}%")
        
        # Get best/worst performers in current phase
        print(f"\n" + "="*70)
        print(f"HISTORICAL BEST PERFORMERS IN '{current_phase}' PHASE")
        print("="*70)
        
        if current_phase in self.performance_df.columns:
            phase_perf = self.performance_df[current_phase].sort_values(ascending=False)
            print("\nTop 5 Sectors (Historical Avg Performance):")
            for i, (sector, perf) in enumerate(phase_perf.head(5).items(), 1):
                cyclicality = self.sensitivity_df.loc[sector, 'Classification']
                print(f"  {i}. {sector}: {perf:.2f}% ({cyclicality})")
            
            print("\nBottom 5 Sectors:")
            for i, (sector, perf) in enumerate(phase_perf.tail(5).items(), 1):
                cyclicality = self.sensitivity_df.loc[sector, 'Classification']
                print(f"  {i}. {sector}: {perf:.2f}% ({cyclicality})")
        
        # Recent momentum
        print(f"\n" + "="*70)
        print(f"RECENT SECTOR MOMENTUM (Last {lookback_years} years)")
        print("="*70)
        
        industry_cols = [col for col in recent_industry.columns if 'Sector Revenue' in col]
        recent_performance = {}
        
        for col in industry_cols:
            sector_name = col.replace('YoY % Change in ', '').replace(' Sector Revenue', '')
            avg_growth = recent_industry[col].mean()
            recent_performance[sector_name] = avg_growth
        
        recent_perf_df = pd.DataFrame.from_dict(recent_performance, orient='index',
                                                columns=['Recent_Avg_Growth'])
        recent_perf_df = recent_perf_df.sort_values('Recent_Avg_Growth', ascending=False)
        
        print("\nTop Performers:")
        print(recent_perf_df.head(5).round(2))
        print("\nBottom Performers:")
        print(recent_perf_df.tail(5).round(2))
        
        # Create recommendations summary
        self._create_recommendation_chart(recent_perf_df, current_phase)
        
        return recent_perf_df
    
    def _create_recommendation_chart(self, recent_perf_df, current_phase):
        """Create visualization of recommendations"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Chart 1: Recent momentum
        colors1 = ['green' if x > 0 else 'red' for x in recent_perf_df['Recent_Avg_Growth']]
        ax1.barh(range(len(recent_perf_df)), recent_perf_df['Recent_Avg_Growth'], 
                color=colors1, alpha=0.7)
        ax1.set_yticks(range(len(recent_perf_df)))
        ax1.set_yticklabels(recent_perf_df.index)
        ax1.set_xlabel('Average Growth Rate (%, Recent Years)', fontsize=11)
        ax1.set_title('Recent Sector Momentum', fontsize=13, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Chart 2: Cyclicality positioning
        sectors_in_both = [s for s in recent_perf_df.index if s in self.sensitivity_df.index]
        
        if sectors_in_both:
            comparison_df = pd.DataFrame({
                'Recent_Performance': [recent_perf_df.loc[s, 'Recent_Avg_Growth'] 
                                      for s in sectors_in_both],
                'GDP_Sensitivity': [self.sensitivity_df.loc[s, 'GDP_Correlation'] 
                                   for s in sectors_in_both]
            }, index=sectors_in_both)
            
            ax2.scatter(comparison_df['GDP_Sensitivity'], 
                       comparison_df['Recent_Performance'],
                       s=100, alpha=0.6, c='blue')
            
            for sector in comparison_df.index:
                ax2.annotate(sector, 
                           (comparison_df.loc[sector, 'GDP_Sensitivity'],
                            comparison_df.loc[sector, 'Recent_Performance']),
                           fontsize=8, alpha=0.7)
            
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('GDP Sensitivity (Cyclicality)', fontsize=11)
            ax2.set_ylabel('Recent Performance', fontsize=11)
            ax2.set_title('Sector Positioning: Cyclicality vs Recent Performance', 
                         fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rotation_recommendations.png', dpi=300, bbox_inches='tight')
        print("\nSaved: rotation_recommendations.png")
        plt.show()


# ============================================================================
# 5. MAIN ANALYSIS RUNNER
# ============================================================================

def run_complete_analysis():
    """
    Run the complete industry rotation and economic cycle analysis
    """
    
    print("="*70)
    print("INDUSTRY ROTATION & ECONOMIC CYCLE ANALYSIS TOOL")
    print("="*70)
    
    # Step 1: Load Data
    print("\nSTEP 1: Loading Data...")
    loader = EconomicDataLoader(EXCEL_FILE, SHEET_NAME)
    loader.load_data().identify_indicators()
    macro_df, industry_df = loader.get_clean_data()
    
    # Step 2: Analyze Economic Cycles
    print("\nSTEP 2: Analyzing Economic Cycles...")
    cycle_analyzer = CycleAnalyzer(macro_df)
    cycle_df = cycle_analyzer.create_cycle_timeline()
    
    # Step 3: Analyze Industry Performance
    print("\nSTEP 3: Analyzing Industry Performance...")
    industry_analyzer = IndustryAnalyzer(macro_df, industry_df)
    
    print("\n3a. Calculating Sector Cyclicality...")
    sensitivity_df = industry_analyzer.analyze_sector_sensitivity()
    
    print("\n3b. Analyzing Performance by Cycle Phase...")
    performance_df = industry_analyzer.analyze_performance_by_cycle(cycle_df)
    
    print("\n3c. Creating Sector Comparison Chart...")
    industry_analyzer.create_sector_comparison_chart()
    
    print("\n3d. Analyzing Sectoral Correlations...")
    correlation_matrix = industry_analyzer.analyze_sectoral_correlations()
    
    print("\n3e. Multi-Factor Sensitivity Analysis...")
    multifactor_sensitivity = industry_analyzer.analyze_multi_factor_sensitivity()
    
    print("\n3f. Sector Volatility Analysis...")
    volatility_df = industry_analyzer.analyze_sector_volatility()
    
    print("\n3g. Sector Momentum Analysis...")
    momentum_df = industry_analyzer.analyze_sector_momentum()
    
    # Step 4: Generate Rotation Recommendations
    print("\nSTEP 4: Generating Rotation Recommendations...")
    recommender = RotationRecommender(macro_df, industry_df, cycle_df, 
                                     performance_df, sensitivity_df)
    recommendations = recommender.get_current_recommendations()
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  1. economic_cycle_timeline.png - Economic cycle visualization")
    print("  2. sector_cyclicality.png - Sector sensitivity to GDP")
    print("  3. performance_by_cycle_phase.png - Heatmap of sector performance")
    print("  4. sector_trends_interactive.html - Interactive sector trends")
    print("  5. sectoral_correlation_matrix.png - Correlation between sectors")
    print("  6. multifactor_sensitivity.png - Sector response to multiple factors")
    print("  7. sector_volatility_analysis.png - Risk-return profiles")
    print("  8. sector_momentum.png - Recent momentum trends")
    print("  9. rotation_recommendations.png - Current recommendations")
    print("\n" + "="*70)
    print("KEY INSIGHTS GENERATED:")
    print("="*70)
    print("✓ Economic cycle identification and timeline")
    print("✓ Sector cyclicality rankings (defensive vs cyclical)")
    print("✓ Historical performance patterns by cycle phase")
    print("✓ Sectoral correlations for diversification insights")
    print("✓ Multi-factor sensitivity (GDP, inflation, credit, IIP)")
    print("✓ Volatility and risk-return analysis")
    print("✓ Momentum analysis (accelerating vs decelerating sectors)")
    print("✓ Current rotation recommendations")
    print("="*70)
    print("\nYou can now use these insights for sector rotation decisions!")
    print("All visualizations are saved in /content/ (check Files panel)")
    print("="*70)


# ============================================================================
# RUN THE ANALYSIS
# ============================================================================

if __name__ == "__main__":
    # Make sure to update EXCEL_FILE variable at the top with your filename!
    run_complete_analysis()
