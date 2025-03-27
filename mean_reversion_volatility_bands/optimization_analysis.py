import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import sys
from scipy import stats
import datetime
from sklearn.preprocessing import MinMaxScaler
import logging
import traceback
import shutil

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
class OptimizationAnalyzer:
    """
    A class for analyzing trading strategy optimization results.
    
    This class provides methods to load, analyze, and visualize optimization
    results from backtesting trading strategies with comprehensive logging
    and error handling.
    """
    
    def __init__(self, optimization_results, output_dir, log_level=logging.INFO):
        """
        Initialize the OptimizationAnalyzer with data path and output directory.
        
        Parameters:
        -----------
        optimization_results : list or DataFrame
            Optimization results to analyze
        output_dir : str, optional
            Directory where analysis results will be saved
        log_level : int, optional
            Logging level (default: logging.INFO)
        """
        # Set up logging
        self.logger = self._setup_logger(log_level)
        
        try:
            # Set default output directory if not provided
            if output_dir is None:
                self.output_dir = "optimization_analysis"
            else:
                self.output_dir = output_dir
                
            # Remove output directory if it exists, then create a new one
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
                self.logger.info(f"Removed existing output directory: {self.output_dir}")
            
            # Create fresh output directory
            os.makedirs(self.output_dir)
            self.logger.info(f"Created new output directory: {self.output_dir}")
            
            # Initialize data attributes
            self.df = optimization_results
            
            
            self.sorted_df = None
            self.correlations = None
            self.best_config = None
            self.avg_return = None
            self.positive_returns = None
            
            # Load data 
            self.load_data()
                       
            self.logger.info(f"Optimization Analyzer initialized. Results will be saved to: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error initializing OptimizationAnalyzer: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
    
    def _setup_logger(self, log_level):
        """
        Set up and configure the logger.
        
        Parameters:
        -----------
        log_level : int
            Logging level
            
        Returns:
        --------
        logger : logging.Logger
            Configured logger
        """
        # Create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Create file handler
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # file_handler = logging.FileHandler(f"optimization_analysis_{timestamp}.log")
        # file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        # file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        # logger.addHandler(file_handler)
        
        return logger
    
    def load_data(self):
        """
        Load optimization data from CSV file.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file with optimization results. If not provided,
            uses the path from initialization.
        
        Returns:
        --------
        self : OptimizationAnalyzer
            Returns self for method chaining
        """
        try:          
            # Validate data structure
            required_columns = ['total_return_pct']
            for col in required_columns:
                if col not in self.df.columns:
                    self.logger.error(f"Required column '{col}' not found in data")
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Clean the DataFrame - remove the first column (usually an index)
            self.df = self.df.copy()
            
            # Check for empty dataframe
            if len(self.df) == 0:
                self.logger.warning("Loaded data is empty")
                
            # Calculate basic statistics
            self._calculate_statistics()
            
            self.logger.info(f"Data loaded successfully: {len(self.df)} configurations")
            return self
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
    
    def _calculate_statistics(self):
        """
        Calculate basic statistics from the optimization data.
        This is an internal method called after loading the data.
        """
        try:
            # Sort by performance
            self.sorted_df = self.df.sort_values('total_return_pct', ascending=False).reset_index(drop=True)
     
            # Calculate correlations
            self.correlations = self.df.corr()['total_return_pct'].drop('total_return_pct').sort_values(ascending=False)
            
            # Find key statistics
            self.best_config = self.sorted_df.iloc[0]
            self.avg_return = self.df['total_return_pct'].mean()
            self.positive_returns = (self.df['total_return_pct'] > 0).sum()
            
            self.logger.debug("Statistics calculated successfully")
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
    
    def _check_data_loaded(self):
        """
        Check if data is loaded before proceeding with analysis.
        
        Raises:
        -------
        ValueError
            If no data is loaded
        """
        if self.df is None:
            self.logger.error("No data loaded")
            raise ValueError("No data loaded. Please load data first with load_data().")
    
    def save_summary_statistics(self):
        """
        Save summary statistics to a text file and print them to console.
        
        Returns:
        --------
        self : OptimizationAnalyzer
            Returns self for method chaining
        """
        try:
            self._check_data_loaded()
            
            summary_path = f"{self.output_dir}/analysis_summary.txt"
            self.logger.info(f"Saving summary statistics to: {summary_path}")

            # Save summary statistics to a text file
            with open(summary_path, "w") as f:
                f.write("========== OPTIMIZATION ANALYSIS ==========\n")
                f.write(f"Number of configurations tested: {len(self.df)}\n")
                f.write(f"Average return: {self.avg_return:.2f}%\n")
                f.write(f"Configurations with positive returns: {self.positive_returns} ({self.positive_returns/len(self.df)*100:.1f}%)\n\n")
                
                f.write("Best configuration:\n")
                for param in self.df.columns:
                    if param != 'total_return_pct':
                        f.write(f"  {param}: {self.best_config[param]}\n")
                f.write(f"  Return: {self.best_config['total_return_pct']:.2f}%\n\n")
                
                f.write("Parameter impact on returns (correlation):\n")
                for param, corr in self.correlations.items():
                    f.write(f"  {param}: {corr:.3f}\n")
                
                f.write("\nOptimal parameter ranges (from top 5 configurations):\n")
                for param in self.df.columns:
                    if param != 'total_return_pct':
                        min_val = self.sorted_df.head(5)[param].min()
                        max_val = self.sorted_df.head(5)[param].max()
                        f.write(f"  {param}: {min_val} to {max_val}\n")
            
                        
            # Save top configurations to CSV
            csv_path = f"{self.output_dir}/top_10_configurations.csv"
            self.sorted_df.head(10).to_csv(csv_path, index=False)
            self.logger.info(f"Top 10 configurations saved to: {csv_path}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error saving summary statistics: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return self
    
    def plot_parameter_correlations(self):
        """
        Create and save a bar plot of parameter correlations with return.
        
        Returns:
        --------
        self : OptimizationAnalyzer
            Returns self for method chaining
        """
        try:
            self._check_data_loaded()
            
            self.logger.info("Creating parameter correlation plot")
            
            plt.figure(figsize=(12, 6))
            self.correlations.plot(kind='bar')
            plt.title('Parameter Correlation with Return', fontsize=14)
            plt.ylabel('Correlation')
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save figure
            fig_path = f"{self.output_dir}/1_parameter_correlations.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            self.logger.info(f"Parameter correlation plot saved to: {fig_path}")
            return self
            
        except Exception as e:
            self.logger.error(f"Error creating parameter correlation plot: {str(e)}")
            self.logger.debug(traceback.format_exc())
            plt.close()  # Ensure figure is closed in case of error
            return self
    
    def plot_top_configurations(self):
        """
        Create and save a text visualization of top configurations.
        
        Returns:
        --------
        self : OptimizationAnalyzer
            Returns self for method chaining
        """
        try:
            self._check_data_loaded()
            
            self.logger.info("Creating top configurations visualization")
            
            plt.figure(figsize=(10, 6))
            plt.axis('off')
            top_table = self.sorted_df.head(5)[['tp_level', 'sl_level', 'max_positions', 'total_return_pct']]
            table_text = []
            table_text.append("TOP 5 CONFIGURATIONS")
            table_text.append("-" * 50)
            for i, row in top_table.iterrows():
                table_text.append(f"Rank {i+1}: Return = {row['total_return_pct']:.2f}%, TP = {row['tp_level']}, " + 
                                 f"SL = {row['sl_level']}, Max Pos = {row['max_positions']}")
            plt.text(0.1, 0.5, '\n'.join(table_text), fontsize=12)
            plt.tight_layout()
            
            # Save figure
            fig_path = f"{self.output_dir}/2_top_configurations.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            self.logger.info(f"Top configurations visualization saved to: {fig_path}")
            return self
            
        except Exception as e:
            self.logger.error(f"Error creating top configurations visualization: {str(e)}")
            self.logger.debug(traceback.format_exc())
            plt.close()  # Ensure figure is closed in case of error
            return self
    
    def plot_parameter_scatter_plots(self, top_n=5):
        """
        Create and save scatter plots for top parameters vs return.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top parameters to plot (default: 5)
            
        Returns:
        --------
        self : OptimizationAnalyzer
            Returns self for method chaining
        """
        try:
            self._check_data_loaded()
            
            # Validate top_n
            if top_n <= 0 or top_n > len(self.correlations):
                self.logger.warning(f"Invalid top_n value ({top_n}). Using min(5, {len(self.correlations)})")
                top_n = min(5, len(self.correlations))
                
            self.logger.info(f"Creating scatter plots for top {top_n} parameters")
            
            for i, param in enumerate(self.correlations.index[:top_n]):

                try:
                    plt.figure(figsize=(8, 6))
                    scatter = plt.scatter(self.df[param], self.df['total_return_pct'], 
                               c=self.df['total_return_pct'], cmap='viridis', alpha=0.8, s=80)
                    plt.title(f'Return vs {param}', fontsize=14)
                    plt.xlabel(param)
                    plt.ylabel('Return %')
                    plt.grid(linestyle='--', alpha=0.7)
                    plt.colorbar(scatter, label='Return %')
                    
                    # Add linear regression line
                    slope, intercept, r_value, p_value, std_err = stats.linregress(self.df[param], self.df['total_return_pct'])
                    x = np.array([min(self.df[param]), max(self.df[param])])
                    y = intercept + slope * x
                    plt.plot(x, y, 'r--', alpha=0.7)
                    plt.text(0.05, 0.95, f'R² = {r_value**2:.3f}', transform=plt.gca().transAxes, 
                            fontsize=10, verticalalignment='top')
                    
                    plt.tight_layout()
                    
                    # Save figure
                    fig_path = f"{self.output_dir}/3_{i+1}_scatter_{param}.png"
                    plt.savefig(fig_path, dpi=300)
                    plt.close()
                    
                    self.logger.info(f"Scatter plot for {param} saved to: {fig_path}")
                    
                except Exception as e:
                    self.logger.error(f"Error creating scatter plot for {param}: {str(e)}")
                    self.logger.debug(traceback.format_exc())
                    plt.close()  # Ensure figure is closed in case of error
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error creating parameter scatter plots: {str(e)}")
            self.logger.debug(traceback.format_exc())
            plt.close()  # Ensure figure is closed in case of error
            return self
    
    def plot_correlation_heatmap(self, top_n=5):
        """
        Create and save a heatmap of correlations between top parameters.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top parameters to include (default: 5)
            
        Returns:
        --------
        self : OptimizationAnalyzer
            Returns self for method chaining
        """
        try:
            self._check_data_loaded()
            
            # Validate top_n
            if top_n <= 0 or top_n > len(self.correlations):
                self.logger.warning(f"Invalid top_n value ({top_n}). Using min(5, {len(self.correlations)})")
                top_n = min(5, len(self.correlations))
                
            self.logger.info(f"Creating correlation heatmap for top {top_n} parameters")
            
            plt.figure(figsize=(10, 8))
            selected_params = self.correlations.index[:top_n]
            corr_matrix = self.df[list(selected_params) + ['total_return_pct']].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Heatmap of Top Parameters', fontsize=14)
            plt.tight_layout()
            
            # Save figure
            fig_path = f"{self.output_dir}/4_correlation_heatmap.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            self.logger.info(f"Correlation heatmap saved to: {fig_path}")
            return self
            
        except Exception as e:
            self.logger.error(f"Error creating correlation heatmap: {str(e)}")
            self.logger.debug(traceback.format_exc())
            plt.close()  # Ensure figure is closed in case of error
            return self
    
    def plot_return_distribution(self):
        """
        Create and save a histogram of return distribution.
        
        Returns:
        --------
        self : OptimizationAnalyzer
            Returns self for method chaining
        """
        try:
            self._check_data_loaded()
            
            self.logger.info("Creating return distribution histogram")
            
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df['total_return_pct'], kde=True)
            plt.title('Distribution of Returns', fontsize=14)
            plt.xlabel('Return %')
            plt.ylabel('Frequency')
            plt.axvline(x=0, color='r', linestyle='--')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save figure
            fig_path = f"{self.output_dir}/5_return_distribution.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            self.logger.info(f"Return distribution histogram saved to: {fig_path}")
            return self
            
        except Exception as e:
            self.logger.error(f"Error creating return distribution histogram: {str(e)}")
            self.logger.debug(traceback.format_exc())
            plt.close()  # Ensure figure is closed in case of error
            return self
    
    def plot_parallel_coordinates(self, top_n=5, bottom_n=5):
        """
        Create and save a parallel coordinates plot comparing top and bottom configurations.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top configurations to include (default: 5)
        bottom_n : int, optional
            Number of bottom configurations to include (default: 5)
            
        Returns:
        --------
        self : OptimizationAnalyzer
            Returns self for method chaining
        """
        try:
            self._check_data_loaded()
            
            # Validate parameters
            if len(self.sorted_df) < top_n + bottom_n:
                self.logger.warning(f"Not enough configurations for top_n={top_n} and bottom_n={bottom_n}. Adjusting values.")
                total = len(self.sorted_df)
                top_n = min(top_n, total // 2)
                bottom_n = min(bottom_n, total // 2)
                
            self.logger.info(f"Creating parallel coordinates plot (top {top_n} vs bottom {bottom_n})")
            
            plt.figure(figsize=(14, 8))
            
            # Select top and bottom configurations
            top = self.sorted_df.head(top_n)
            bottom = self.sorted_df.tail(bottom_n)
            combined = pd.concat([top, bottom])
            
            # Normalize the data for parallel coordinates
            numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
            
            # Check if total_return_pct is in numeric_cols before removing
            if 'total_return_pct' in numeric_cols:
                numeric_cols.remove('total_return_pct')  # Don't scale the target variable
            
            scaler = MinMaxScaler()
            combined_scaled = combined.copy()
            
            # Handle case with no numeric columns
            if len(numeric_cols) > 0:
                combined_scaled[numeric_cols] = scaler.fit_transform(combined[numeric_cols])
            else:
                self.logger.warning("No numeric columns found for parallel coordinates plot")
            
            # Add a column to indicate top or bottom
            combined_scaled['group'] = ['Top' if i < top_n else 'Bottom' for i in range(len(combined_scaled))]
            
            # Plot parallel coordinates
            pd.plotting.parallel_coordinates(combined_scaled, 'group', color=['#1f77b4', '#d62728'])
            plt.title('Parallel Coordinates Plot: Top vs Bottom Configurations', fontsize=14)
            plt.grid(linestyle='--', alpha=0.7)
            plt.legend(loc='upper right')
            plt.tight_layout()
            
            # Save figure
            fig_path = f"{self.output_dir}/6_parallel_coordinates.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            self.logger.info(f"Parallel coordinates plot saved to: {fig_path}")
            return self
            
        except Exception as e:
            self.logger.error(f"Error creating parallel coordinates plot: {str(e)}")
            self.logger.debug(traceback.format_exc())
            plt.close()  # Ensure figure is closed in case of error
            return self
    
    def plot_pairplot(self, top_n=3):
        """
        Create and save a pairplot of top parameters.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top parameters to include (default: 3)
            
        Returns:
        --------
        self : OptimizationAnalyzer
            Returns self for method chaining
        """
        try:
            self._check_data_loaded()
            
            # Validate top_n
            if top_n <= 0 or top_n > len(self.correlations):
                self.logger.warning(f"Invalid top_n value ({top_n}). Using min(3, {len(self.correlations)})")
                top_n = min(3, len(self.correlations))
                
            self.logger.info(f"Creating pairplot for top {top_n} parameters")
            
            # Create a categorical column for positive/negative returns
            self.df['return_category'] = np.where(self.df['total_return_pct'] > 0, 'Positive', 'Negative')
            
            top_params = self.correlations.index[:top_n]
            
            # Validate that we have parameters
            if len(top_params) == 0:
                self.logger.warning("No parameters found for pairplot")
                return self
                
            pair_df = self.df[list(top_params) + ['total_return_pct', 'return_category']]
            
            # Create and save the pairplot
            pair_plot = sns.pairplot(pair_df, vars=list(top_params) + ['total_return_pct'], 
                         hue='return_category',
                         palette={'Positive': 'green', 'Negative': 'red'},
                         plot_kws={'alpha': 0.6, 's': 80})
                         
            pair_plot.fig.suptitle('Pairplot of Top Parameters', y=1.02, fontsize=16)
            plt.tight_layout()
            
            # Save figure
            fig_path = f"{self.output_dir}/7_pairplot.png"
            plt.savefig(fig_path, dpi=300)
            plt.close()
            
            self.logger.info(f"Pairplot saved to: {fig_path}")
            return self
            
        except Exception as e:
            self.logger.error(f"Error creating pairplot: {str(e)}")
            self.logger.debug(traceback.format_exc())
            plt.close()  # Ensure figure is closed in case of error
            return self
    
    def run_all_analyses(self):
        """
        Run all analysis and visualization methods in sequence.
        
        Returns:
        --------
        self : OptimizationAnalyzer
            Returns self for method chaining
        """
        self.logger.info("Starting full analysis pipeline")
        
        # Create a list of all analysis methods
        analysis_methods = [
            self.save_summary_statistics,
            self.plot_parameter_correlations,
            self.plot_top_configurations,
            self.plot_parameter_scatter_plots,
            self.plot_correlation_heatmap,
            self.plot_return_distribution,
            self.plot_parallel_coordinates,
            self.plot_pairplot
        ]
        
        # Run each method, continue even if one fails
        for method in analysis_methods:
            try:
                method()
            except Exception as e:
                self.logger.error(f"Error in {method.__name__}: {str(e)}")
                self.logger.debug(traceback.format_exc())
                # Continue with other methods even if one fails
        
        self.logger.info(f"Analysis pipeline completed. Results saved to {self.output_dir}/")
        return self
