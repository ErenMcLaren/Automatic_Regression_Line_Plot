class RegressionLine:
    def __init__(self, x_data, y_data, specified_plot_name = "regression_plot", verbose = False, debug = False):
        
            # Programmatic Info:
        self.verbose = verbose
        self.debug   = debug
        
            # Stats on Data:
        self.x_data              = x_data
        self.y_data              = y_data
        self.sample_size         = None
        self.degrees_of_freedom  = None
        self.x_data_mean         = None
        self.y_data_mean         = None
        self.x_sample_variance   = None
        self.y_sample_variance   = None
        
            # Regression Line computation:
        self.regression_line_slope      = None
        self.regression_line_y_int      = None
        self.sxx                        = None
        self.sxy                        = None
        self.residuals                  = None
        self.residuals_squared          = None
        self.rss                        = None
        self.s_squared_sample_variance  = None
        
            # Hypothesis testing Regression Line Slope:
        self.estimated_standard_error_in_regression_line_slope  = None
        self.regression_line_slope_test_statistic               = None
        self.p_value_for_slope                                  = None
        self.p_value_for_slope_error                            = None
        self.regression_line_slope_lower_bound                  = {}
        self.regression_line_slope_upper_bound                  = {}
        
            # Hypothesis testing Regression Line Y-intercept:
        self.estimated_standard_error_in_regression_line_y_int  = None
        self.regression_line_y_int_test_statistic               = None
        self.p_value_for_y_int                                  = None
        self.p_value_for_y_int_error                            = None
        self.regression_line_y_int_lower_bound                  = {}
        self.regression_line_y_int_upper_bound                  = {}
        
            # Plot Data:
        self.x_first_quarter            = None
        self.y_first_quarter            = None
        self.x_third_quarter            = None
        self.y_third_quarter            = None
        self.equation_text_coordinates  = None
        
            # Plot Sesthetic Properties:
        self.specified_plot_name   = specified_plot_name
        self._font_family          = 'serif'
        self._plot_background      = "dark_background"
        self.plot_title            = "Computed Linear Regresion of Y on X"
        self.x_label               = "X"
        self.y_label               = "Y"
        self.graph_axes_label_size = 16
        self.formatted_confidence_interval_grid_table = None
        
            # Table Properties:
        self._column_colors = "#000000"
        self._cell_colors   = "#000000"
        self._only_fitting_table_font_size = 6.5
        
            # Scatterplot Properties
        self._scatter_plot_dots_color    = "#a7cafc"
        self._scatter_plot_dots_size     = 10
        
            # Regression Line Properties:
        self._regression_line_fontsize = 7
        self._regression_line_plot_color = "#fcba03"
        self._regression_line_thickness  = 1
        
            # Figure Properties:
        self._default_dots_per_inch = 300
        
            # Student T-Dist Values:
            # http://statcalculators.com/students-t-distribution-table/
        self.t_values_for_two_tailed_alpha = {
            80: 1.3304,
            90: 1.7341,
            95: 2.1009,
            98: 2.5524,
            99: 2.8784,
            99.9: 3.9216
        }
    
        
    def compute_sample_size(self):
        """
        Compute the sample size of the sample.
        Compares the number of elements along each axis 
        and determines if it matches or does not.
        If it matches: set the sample size (using the y_data)
        If it doesn't: throw an error
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the sample size.")
        try:
            if self.x_data is not None and self.y_data is not None:
                computed_sample_size_from_x_data = len(self.x_data)
                computed_sample_size_from_y_data = len(self.y_data)
                try:
                    if computed_sample_size_from_x_data is computed_sample_size_from_y_data:
                        self.sample_size = computed_sample_size_from_y_data
                except:
                    print("Length of x and y data points don't match.")
        except TypeError:
            print("There is either no input x or y data or a non-numerical value exists in x or y data preventing numerical operations")
        return self
    
    def compute_degrees_of_freedom(self):
        """
        Compute the degrees of freedom for use in
        the Student t-distribution
        IMPORTANT: given that we assume the regression
        of Y on X is linear, i.e.,
        E(Y|X = x_{i}) = \beta_{0} + \beta_{1} x_{1} 
        the degrees of freedom is n - 2 because
        we are estimating two parameters,
        \beta_{0} and \beta_{1}
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the degrees of freedom.")
        try:
            if self.degrees_of_freedom is None:
                self.degrees_of_freedom = self.sample_size - 2
            else:
                print("Error calculating degrees of freedom > missing the sample size. Either the chronology of the program messed up or something's very wrong.")
        except TypeError:
            print("Error calculating degrees of freedom: non-numerical value for the sample size preventing numerical operations")
        return self
        
    def compute_sample_x_mean(self):
        """
        Compute the mean along the x-axis of the sample data
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the mean of the sample data across the x-axis.")
        try: 
            if self.x_data is not None:
                self.x_data_mean = sum(self.x_data) / len(self.x_data)
            else:
                print("Error calculating mean along x-axis > there is NO X-DATA.")
        except TypeError:
            print("Error calculating sample X mean: non-numerical (not an int, float) element in X data values when calculating sample X mean preventing numerical operations.")
        return self
    
    def compute_sample_y_mean(self):
        """
        Compute the mean along the y-axis of the sample data
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the mean of the sample data across the y-axis.")
        try:
            if self.y_data is not None:
                self.y_data_mean = sum(self.y_data) / len(self.y_data)
            else:
                print("Error calculating mean along y-axis > there is NO Y-DATA.")
        except TypeError:
            print("Error calculating sample Y mean: non-numerical (not an int, float) element in Y data values when calculating sample Y mean preventing numerical operations.")
        return self
    
    def compute_sample_x_variance(self):
        """
        Compute the X SAMPLE variance
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the variance of the sample data across the x-axis.")
        try:
            if self.x_sample_variance is None:
                self.x_sample_variance = sum([(i - self.sample_x_mean(x_data))**2 for i in self.x_data]) / (len(x_data) - 1)
            else:
                print("Error calculating X-sample variance > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except TypeError:
            print("Error calculating sample X-variance: non-numerical (not an int, float) element in X data values when calculating sample X variance preventing numerical operations.")
        return self
    
    def compute_sample_y_variance(self):
        """
        Compute the Y SAMPLE variance
        """
        if self.debug or self.debug:
            print(f"Debugging > Currently computing the variance of the sample data across the y-axis.")
        try:
            if self.y_sample_variance is None:
                self.y_sample_variance = sum([(i - self.sample_y_mean(y_data))**2 for i in self.y_data]) / (len(y_data) - 1)
            else:
                print("Error calculating Y-sample variance > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating sample Y-variance: non-numerical (not an int, float) element in Y data values when calculating sample Y variance preventing numerical operations.")
        return self
    
    def calculate_sxx(self):
        """
        Compute the sum of (x_i - x_mean) times (x_i - x_mean)
        > S = sum
        > X = (x_i - x_mean)
        > SXX = sum of (x_i - x_mean) times (x_i - x_mean)
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing SXX, the sum of (x_i - x_mean) times (x_i - x_mean).")
        try:
            if self.sxx is None:
                self.sxx = sum(a * b for a, b in zip([(x_i - self.x_data_mean) for x_i in self.x_data], [(x_i - self.x_data_mean) for x_i in self.x_data]))
            else:
                print("Error calculating SXX > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating SXX: non-numerical element (not an int or float) in X data OR the mean of X data is non-numerical preventing numerical operations.")
        return self
    
    def calculate_sxy(self):
        """
        Compute the sum of (x_i - x_mean) times (y_i - y_mean)
        > S = sum
        > Y = (y_i - y_mean)
        > SXY = sum of (x_i - x_mean) times (y_i - y_mean)
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing SXY, the sum of (x_i - x_mean) times (y_i - y_mean).")
        try:
            if self.sxy is None:
                self.sxy = sum(a * b for a, b in zip([x_i - self.x_data_mean for x_i in self.x_data], [y_i - self.y_data_mean for y_i in self.y_data]))
            else:
                print("Error calculating SXY > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating SXY > non-numerical element (not an int or float) in X data OR the mean of X data -OR- Y data OR the mean of Y data is non-numerical preventing numerical operations.")
        return self
    
    def compute_regression_line_slope(self):
        """
        Compute the slope of the regression line with
        beta_{0} = SXY / SXX
        """
        if self.debug:
            print(f"Debugging > Currently computing the regression line slope.")
        try:
            if self.regression_line_slope is None:
                self.regression_line_slope = self.sxy / self.sxx
            else:
                print("Error calculating regression line slope > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating regression line slope > non-numerical value for SXX OR SXY preventing numerical operations.")
        return self
    
    def compute_regression_line_y_int(self):
        """
        Compute the y-intercept of the regression line with
        beta_{1} = y_mean - (beta_{0} * x_mean)
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the regression line y-intercept.")
        try:
            if self.regression_line_y_int is None:
                self.regression_line_y_int = self.y_data_mean - (self.regression_line_slope * self.x_data_mean)
            else:
                print("Error calculating regression line y-intercept > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating regression line y-intercept > non-numerical X data mean OR Y data mean (not an int or float)")
        return self
    
    def compute_residuals(self):
        """
        Compute the residuals using the regression line with
        \hat{e} = \sum{y_data_i - y_regline_i}
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the residuals (e) of the regression lines.")
        try:
            if self.residuals is None:
                self.residuals = [(self.y_data[index] - self.regression_line_at_x(x_i))for index, x_i in enumerate(self.x_data)]
            else:
                print("Error calculating regression line residuals > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating regression line residuals > non-numerical value (not an int or float) present in X or Y data OR, more likely, the computed regression line evaluated at x threw non-numerical somehow.")
        return self
    
    def compute_residuals_squared(self):
        """
        Compute the residuals squared by squaring the residuals:
        \hat{e_i}^{2} = \hat{e_i} * \hat{e_i}
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently squaring the residuals (e) of the regression lines.")
        try:
            if self.residuals_squared is None:
                self.residuals_squared = [x * y for x, y in zip(self.residuals, self.residuals)]
            else:
                print("Error calculating regression line residuals squared > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating regression line residuals squared > non-numerical value (not an int or float) present in original list of residuals.")
        return self
    
    def compute_rss(self):
        """
        Compute the sum of the residuals squared (RSS) using
        RSS = \sum{\hat{e_{i}^{2}}}
        > R = residuals
        > S = sum
        > S = squared
        > RSS = residuals sum squared
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the RSS, sum of the residuals squared.")
        try:
            if self.rss is None:
                self.rss = sum(self.residuals_squared)
            else:
                print("Error calculating RSS > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating RSS > non-numerical value (not an int or float) present in original list of residuals squared.")
        return self
    
    def sample_variance(self):
        """
        Compute the SAMPLE variance using
        S^{2} = RSS / (n - 2)*
        *divisor n - 2 is because we have two estimated
        parameters, \beta_{0} and \beta_{1}
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the sample variance with RSS.")
        try:
            if self.s_squared_sample_variance is None:
                self.s_squared_sample_variance = self.rss / (self.sample_size - 2)
            else:
                print("Error calculating sample variance with RSS > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating sample variance with RSS > non-numerical value (not an int or float) for RSS or sample size.")
        return self
    
    def sample_standard_deviation(self):
        """
        Compute the SAMPLE standard deviation by square 
        rooting the sample variance by:
        S = \sqrt{S^{2}}
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the sample standard deviation.")
        try:
            if self.s_sample_standard_deviation is None:
                self.s_sample_standard_deviation = np.sqrt(self.s_squared_sample_variance)
            else:
                print("Error calculating sample std. dev with S squared > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating sample std. dev with S squared > non-numerical value (not an int or float) for std. dev with S squared")
            
        return self
    
    def compute_estimated_standard_error_in_regression_line_slope(self):
        """
        Compute the estimated standard error of 
        the slope of the regression line, \hat{\beta_{1}}
        se(\hat{\beta_{1}}) = \frac{S}{\sqrt{SXX}}
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the estimated standard error of the slope of the regression line.")
        try:
            if self.estimated_standard_error_in_regression_line_slope is None:
                self.estimated_standard_error_in_regression_line_slope = np.sqrt((self.s_squared_sample_variance / self.sxx))
            else:
                print("Error calculating estimated standard error in regression line slope > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating estimated standard error in regression line slope > non-numerical value (not an int or float) for std. dev with S squared OR SXX.")
        return self
    
    def compute_t_statistic_for_regression_line_slope(self):
        """
        Compute the test statistic for use in the T-score
        distribution, calculated in this case where
        H_{0}: \beta_{0} = 0, H_{1}: \beta_{0} \neq 0, 
        T = \frac{\hat{\beta_{1}} - 0}{se(\hat{\beta_{1}})}
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the test statistic for use in the T-score distribution of the slope of the regression line.")
        try:
            if self.regression_line_slope_test_statistic is None:
                self.regression_line_slope_test_statistic = self.regression_line_slope / self.estimated_standard_error_in_regression_line_slope
            else:
                print("Error calculating T-statistic for regression line slope > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating T-statistic for regression line slope > non-numerical value (not an int or float) for regression line slope OR estimated standard error in the slope.")
        return self
    
    def compute_p_value_given_test_statistic_for_slope(self):
        """
        Compute the p-value for the test statistic
        GIVEN that we are testing hypotheses in the form:
        H_{0}: \beta_{0} = 0, H_{1}: \beta_{0} \neq 0
        This requires performing an integral over the 
        Student T-distribution
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the p-value of the slope of the regression line.")
        try:
            integral = integrate.quad(self.student_t_distribution_at_x, -1 * self.regression_line_slope_test_statistic, self.regression_line_slope_test_statistic)
            self.p_value_for_slope, self.p_value_for_slope_error = 1 - integral[0], integral[1]
        except ValueError:
            print("Error calculating the p-value for the test statistic for regression line slope > integral exploded or something.")
        return self
    
    def compute_confidence_interval_for_slope(self):
        """
        Compute a variety of confidence intervals
        using the Student t-scores with distribution
        T(alpha, dof = 18) for the true value
        of the slope for the linear regression
        of Y on X.
        """
        if self.debug or self.verbose:
            print(f"Currently computing the upper and lower values of the slope of the regression line per several confidence intervals.")
        try:
            for index in range(len(self.t_values_for_two_tailed_alpha)):
                confidence_interval_percentage = list(self.t_values_for_two_tailed_alpha.keys())[index]
                current_t_value = list(self.t_values_for_two_tailed_alpha.values())[index]
                if self.debug or self.debug:
                    print(f"Debugging > Currently using CI {confidence_interval_percentage}% using T-value: {current_t_value}")
                current_calculated_regression_line_slope_lower_bound = self.regression_line_slope - (current_t_value * self.estimated_standard_error_in_regression_line_slope)
                current_calculated_regression_line_slope_upper_bound = self.regression_line_slope + (current_t_value * self.estimated_standard_error_in_regression_line_slope)
                if self.debug or self.debug:
                    print(f"Debugging > Determined slope to be ({current_calculated_regression_line_slope_upper_bound}, {current_calculated_regression_line_slope_lower_bound})")
                self.regression_line_slope_upper_bound[index] = current_calculated_regression_line_slope_upper_bound
                self.regression_line_slope_lower_bound[index] = current_calculated_regression_line_slope_lower_bound
        except ValueError:
            print("Error calculating the CIs for the regression line slope > an non-numeric value arose during the calculation of multiple CIs for the regression line slope. Try running this again, or there's something terribly wrong.")
        return self
    
    def compute_estimated_standard_error_in_regression_line_y_int(self):
        """
        Compute the estimated standard error of 
        the y-intercept of the regression line, \hat{\beta_{0}}
        se(\hat{\beta_{0}}) = S\sqrt{\frac{1}{n} + \frac{\bar{x}^{2}}{SXX}}
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the estimated standard error of the y-intercept of the regression line.")
        try:
            if self.estimated_standard_error_in_regression_line_y_int is None:
                self.estimated_standard_error_in_regression_line_y_int = np.sqrt(self.s_squared_sample_variance) * np.sqrt((1 / self.sample_size) + ((self.x_data_mean**2) / self.sxx))
            else:
                print("Error calculating estimated standard error in regression line y-intercept > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating estimated standard error in regression line y-intercept > non-numerical value (not an int or float) for sample size, mean of X data, SXX, or S squared sample variance.")
        return self
    
    def compute_t_statistic_for_regression_line_y_int(self):
        """
        Compute the test statistic for use in the T-score
        distribution, calculated in this case where
        H_{0}: \beta_{0} = 0, H_{1}: \beta_{0} \neq 0, 
        T = \frac{\hat{\beta_{1}} - 0}{se(\hat{\beta_{1}})}
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the test statistic for use in the T-score distribution of the y-intercept of the regression line.")
        try:
            if self.regression_line_y_int_test_statistic is None:
                self.regression_line_y_int_test_statistic = self.regression_line_y_int / self.estimated_standard_error_in_regression_line_y_int
            else:
                print("Error calculating test statistic for use in the T-score distribution of the regression line y-intercept > value already exists, meaning the chronology of this program messed up. Either try running it again or something's very wrong.")
        except ValueError:
            print("Error calculating etest statistic for use in the T-score distribution of the regression line y-intercept > non-numerical value (not an int or float) for the regression line y-intercept OR estimated standard error in the y-intercept.")
        return self
    
    def compute_p_value_given_test_statistic_for_y_int(self):
        """
        Compute the p-value for the test statistic
        GIVEN that we are testing hypotheses in the form:
        H_{0}: \beta_{1} = 0, H_{1}: \beta_{1} \neq 0
        This requires performing an integral over the 
        Student T-distribution
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the p-value of the y-intercept of the regression line.")
        try:
            integral = integrate.quad(self.student_t_distribution_at_x, -1 * self.regression_line_y_int_test_statistic, self.regression_line_y_int_test_statistic)
            self.p_value_for_y_int, self.p_value_for_y_int_error = 1 - integral[0], integral[1]
        except ValueError:
            print("Error calculating the p-value for the test statistic for regression line y-intercept > integral exploded or something.")
        return self
    
    def compute_confidence_interval_for_y_int(self):
        """
        Compute a variety of confidence intervals
        using the Student t-scores with distribution
        T(alpha, dof = 18) for the true value
        of the y-intercept for the linear regression
        of Y on X.
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the upper and lower values of the y-intercept of the regression line per several confidence intervals.")
        try:
            for index in range(len(self.t_values_for_two_tailed_alpha)):
                confidence_interval_percentage = list(self.t_values_for_two_tailed_alpha.keys())[index]
                current_t_value = list(self.t_values_for_two_tailed_alpha.values())[index]
                if self.debug or self.verbose:
                    print(f"Debugging > Currently using CI {confidence_interval_percentage}% using T-value: {current_t_value}")
                current_calculated_regression_line_y_int_lower_bound = self.regression_line_y_int - (current_t_value * self.estimated_standard_error_in_regression_line_y_int)
                current_calculated_regression_line_y_int_upper_bound = self.regression_line_y_int + (current_t_value * self.estimated_standard_error_in_regression_line_y_int)
                if self.debug or self.verbose:
                    print(f"Debugging > Determined y-int to be ({current_calculated_regression_line_y_int_upper_bound}, {current_calculated_regression_line_y_int_lower_bound})")
                self.regression_line_y_int_upper_bound[index] = current_calculated_regression_line_y_int_upper_bound
                self.regression_line_y_int_lower_bound[index] = current_calculated_regression_line_y_int_lower_bound
        except ValueError:
            print("Error calculating the CIs for the regression line slope > an non-numeric value arose during the calculation of multiple CIs for the regression line y-intercept. Try running this again, or there's something terribly wrong.")
        return self
    
    def student_t_distribution_at_x(self, x):
        """
        The Student t-distribution (distribution
        given by degrees of freedom ALREADY COMPUTED 
        FROM SAMPLE DATA - this dist. is NOT general)
        in functional form.
        
        Callable function, for use in integration to
        obtain the p value
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently computing the value of the Student T Distribution at {x}.")
        return (gamma((self.degrees_of_freedom + 1) / 2) * ((1 + ((x ** 2) / (self.degrees_of_freedom))) ** (-1 * ((self.degrees_of_freedom + 1 ) / 2))) / (np.sqrt(np.pi * self.degrees_of_freedom) * gamma(self.degrees_of_freedom / 2)))
    
    def regression_line_at_x(self, x):
        """
        The functional form of the computed
        regression line.
        
        Callable function, for regression line
        at x:
        """
        if self.debug or self.verbose:
                print(f"Debugging > Currently evaluating the regression line with {x}")
        return (self.regression_line_slope * x) + self.regression_line_y_int
    
    def get_regression_line_statistical_summary(self):
        """
        Print out the numbers characterizing the
        regression line, including hypothesis 
        testing the slope and y-intercept
        """
        if self.verbose or self.debug:
            print(f"Debugging > Currently obtaining a statistical summary of the data")
            self.format_confidence_interval_table_data()
            statistical_summary = f"""
            #1 : General Statistics of Sample:
            Sample Size     = {self.sample_size}
            X-Data Mean     = {self.x_data_mean}
            Y-Data Mean     = {self.y_data_mean}
            Sample Variance = {self.s_squared_sample_variance}

            #2 : Intermediate Calculations:
            SXX             = {self.sxx}
            SXY             = {self.sxy}

            #3 : Computed Regression Line:
            slope           = {self.regression_line_slope}
            y-intercept     = {self.regression_line_y_int}

            #4 :  Residual Calculations:
            n(e)            = {len(self.residuals)}
            n(e^2)          = {len(self.residuals_squared)}
            RSS             = {self.rss}

            #5 : Hypothesis Testing:
            Standard Error in Regression Line Slope = {self.estimated_standard_error_in_regression_line_slope}
            Testing hypotheses:
            H0: regression line slope = 0
            H1: regression line slope =/= 0
            T = {self.regression_line_slope_test_statistic}
            p = {self.p_value_for_slope} ± {self.p_value_for_slope_error}

            #6 : Hypothesis Testing:
            Standard Error in Regression Line Y-Intercept = {self.estimated_standard_error_in_regression_line_y_int}
            Testing hypotheses:
            H0: regression line y-intercept = 0
            H1: regression line y-intercept =/= 0
            T = {self.regression_line_y_int_test_statistic}
            p = {self.p_value_for_y_int} ± {self.p_value_for_y_int}
            
            Confidence Intervals: {self.formatted_confidence_interval_grid_table}
            """
            print(statistical_summary)
        return self
    
    def find_central_coordinates_for_each_quadrant(self):
        """
        When constructing the plot and displaying the
        regression line formula, the formula must not
        overlap on many data points. We use this function
        to perform the first step in finding a 'sparce'
        region of the to-be plot.
        
        To do so, we find the central coordinates of the
        to-be plot by finding the data 'midpoint' by
        calculating average of the maximum and minimum 
        data value along an axis. 
        
        Then, calculate the midpoint of the midpoint
        by using both the minimum and maximum data value
        along an axis again. We call these midpoints
        of midpoints 'quarters' for some reason. 
        
        This produces four tuples:
        # quarter_1: (x_quarter_1, Y_quarter_1)
        # quarter_2: (x_quarter_1, Y_quarter_3)
        # quarter_3: (x_quarter_3, Y_quarter_1)
        # quarter_4: (x_quarter_3, Y_quarter_3)
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently finding the central coordinates in each quadrant.")
        try:
            x_first_quarter = (np.min(self.x_data) + ((np.max(self.x_data) + np.min(self.x_data)) / 2)) / 2
            y_first_quarter = (np.min(self.y_data) + ((np.max(self.y_data) + np.min(self.y_data)) / 2)) / 2
            x_third_quarter = (np.max(self.x_data) + ((np.max(self.x_data) + np.min(self.x_data)) / 2)) / 2
            y_third_quarter = (np.max(self.y_data) + ((np.max(self.y_data) + np.min(self.y_data)) / 2)) / 2
            self.x_first_quarter = x_first_quarter
            self.y_first_quarter = y_first_quarter
            self.x_third_quarter = x_third_quarter
            self.y_third_quarter = y_third_quarter
        except Exception as e:
            print(f"Error finding the four quadrants' midpoints > {e}")
        return self
    
    def determine_least_dense_quadrant_in_plot(self):
        """
        With this function, we determine the least
        dense quadrant (quadrant of plot that has 
        least amount of data points).
        We do this by using the central point
        P_{q} = (x_{q}, y_{q}) characterizing each quadrant 
        as computed before, and then, using several 
        values of fixed radii (R_{j}) with point P_{i}:
        
        1. if |x_{i} - x_{q}| > R_{j}, reject point P_{i}.
        2. if |y_{i} - y_{q}| > R_{j}, reject point P_{i}.
        3. if Euclidean distance between P_{i} and P_{q} > R{j},
        reject P_{i}.
        4. Finally, keep P_{i}.
        
        The quadrant with the least number of P_{i}'s is 
        the "least dense".
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently finding a relatively empty quadrant in the plot to place the regression line equation.")
        try:
            quadrant_central_points = [
            (self.x_first_quarter, self.y_first_quarter), # quarter_1: (x_quarter_1, Y_quarter_1)
            (self.x_first_quarter, self.y_third_quarter), # quarter_2: (x_quarter_1, Y_quarter_3)
            (self.x_third_quarter, self.y_first_quarter), # quarter_3: (x_quarter_3, Y_quarter_1)
            (self.x_third_quarter, self.y_third_quarter)  # quarter_4: (x_quarter_3, Y_quarter_3)
            ]
            radii_to_check = [
                10, 
                20, 
                30
            ]
            number_of_data_points_in_ith_quadrant = [0, 0, 0, 0]
            for numbered_point_index, quadrant_central_point in enumerate(quadrant_central_points):
                for radius in radii_to_check:
                    if self.debug:
                        print(f"Debugging > Currently checking {quadrant_central_point} with radius {radius}")
                    for index in range(len(self.x_data)):
                        if not (self.calculate_linear_distance(self.x_data[index], quadrant_central_point[0]) > radius):
                            if not (self.calculate_linear_distance(self.y_data[index], quadrant_central_point[1]) > radius):
                                if not (self.calculate_euclidean_distance((self.x_data[index], self.y_data[index]),(quadrant_central_point[0], quadrant_central_point[1])) > radius):
                                    if self.debug:
                                        print(f"Debugging > Found data within radius of {radius} data-units: ({self.x_data[index]}, {self.y_data[index]})")
                                    number_of_data_points_in_ith_quadrant[numbered_point_index] += 1
                minimum_number_of_points = np.min(number_of_data_points_in_ith_quadrant)
                if number_of_data_points_in_ith_quadrant.count(minimum_number_of_points) > 1:
                    index_where_minimum_number_occurs = [index for index, points in enumerate(number_of_data_points_in_ith_quadrant) if points == min(number_of_data_points_in_ith_quadrant)][0]
                if number_of_data_points_in_ith_quadrant.count(minimum_number_of_points) < 2:
                    index_where_minimum_number_occurs = number_of_data_points_in_ith_quadrant.index(minimum_number_of_points)
                if self.debug:
                    print(f"Debugging > Minimum number of data points in quadrant {index_where_minimum_number_occurs + 1} with radius {radius}")
            self.equation_text_coordinates = quadrant_central_points[index_where_minimum_number_occurs]
        except Exception as e:
            print(f"Error when finding the least dense quadrant in the plot > {e}")
        return self
    
    def calculate_linear_distance(self, point_0, point_1):
        """
        A helper function that computes
        |p_{1} - p_{0}| = L = linear_distance.
        Along a single axis, of course.
        """
        if self.debug or self.verbose:
            (f"Debugging > Calculating the linear distance between two points: |{point_0} - {point_1}| = L.")
        return np.abs((point_1 - point_0))
    
    def calculate_euclidean_distance(self, point_0, point_1):
        """
        A helper function that computes the 
        Euclidean distance between two points:
        d = \sqrt{\left( {x_1 - x_2 } \right)^2 + \left( {y_1 - y_2 } \right)^2}
        (# https://equplus.net/eqninfo/Equation-334.html)
        """
        if self.debug or self.verbose:
            (f"Debugging > Calculating the Euclidean distance between two points: sqrt({(point_1[0] - point_0[0])}^2 + {point_1[1] - point_0[1]}^2) = d.")
        return np.sqrt((point_1[0] - point_0[0])**2 + (point_1[1] - point_0[1])**2)
    
    def round_to_n(self, x, n):
        """
        A helper function that rounds number x
        to the nth numerical place.
        If x = 3356.4, n = 3, 
        then n is the hundreds place.
        """
        if self.debug or self.verbose:
            (f"Debugging > Currently rounding {x} to {n} numerical place.")
        return round(x, -int(np.floor(np.log10(x))) + (n - 1))
    
    def obtain_iterable_range_for_x_axis(self):
        """
        A helper function that produces, using 
        the raw data, an iterable data type for 
        use in plotting the regression line.
        """
        if self.debug or self.verbose:
            (f"Debugging > Currently obtaining an interable range along the x-axis for use in the plot.")
        return np.arange(min(self.x_data), max(self.x_data))
    
    def validate_regression_line_plot_settings(self):
        """
        Quickly validate the choice of Matplotlib style
        even though it's 99% a good idea to use "dark_background".
        Dunno why this is even here.
        """
        if self.debug or self.verbose:
            print(f"Debugging > Currently validating plot aesthetic choices.")
        if self._plot_background not in plt.style.available:
            raise Exception("""
            Your desired plot style is not included in the list of 
            available Matplotlib styles. It is HIGHLY recommended to
            use "dark_background" with this plot style. Please run 
            >>> plt.style.available
            to see the whole list, or reference the list below:
            ['grayscale',
             'seaborn-notebook',
             'tableau-colorblind10',
             'seaborn-white',
             'seaborn-paper',
             'fast',
             'seaborn-deep',
             'seaborn-dark-palette',
             'bmh',
             'fivethirtyeight',
             '_classic_test',
             'classic',
             'Solarize_Light2',
             'seaborn-whitegrid',
             'seaborn',
             'seaborn-poster',
             'dark_background',
             'seaborn-muted',
             'seaborn-colorblind',
             'seaborn-talk',
             'seaborn-pastel',
             'seaborn-bright',
             'seaborn-ticks',
             'seaborn-darkgrid',
             'ggplot',
             'seaborn-dark']
            """)
        return self
    
    def format_confidence_interval_table_data(self):
        """
        Put all the values of the regression line slope and 
        y-intercept into a neatly-prepared text format for
        us in logging out/printing and for use in the 
        plot.
        """
        if self.debug or self.verbose:
            print("Debugging > Currently packaging all CI values into presentable text.")
        try:
            if self.regression_line_slope_lower_bound is not None:
                data = []
                for key in self.regression_line_slope_lower_bound.keys():
                    item_1_ci_percentage = f"{list(self.t_values_for_two_tailed_alpha.keys())[key]}%"
                    item_2_slope_value   = f"({self.round_to_n(self.regression_line_slope_upper_bound[key], 5)}, {self.round_to_n(self.regression_line_slope_lower_bound[key], 5)})"
                    item_3_y_int_value   = f"({self.round_to_n(self.regression_line_y_int_upper_bound[key], 5)}, {self.round_to_n(self.regression_line_y_int_lower_bound[key], 5)})"
                    data.append([item_1_ci_percentage, item_2_slope_value, item_3_y_int_value])
                self.formatted_confidence_interval_grid_table = data
            else:
                print(f"Error packaging all CI values into presentable text > the program has yet to perform CI computation for the regression line's slope and y-intercept")
        except Exception as e:
            print(f"Error packaging all CI values into presentable text > {e}")
        return self
    
    def format_latex_regression_line_equation(self, rounding_to = 5):
        if self.debug or self.verbose:
            print("Debugging > Currently formatting the regression line equation.")
        first_part  = f"$\hat{{y}} = {self.round_to_n(self.regression_line_slope, rounding_to)}x"
        second_part = f"+ {self.round_to_n(self.regression_line_y_int, rounding_to)}$"
        return (first_part + second_part)
    
    def construct_regression_line_plot(self):
        """
        Construct the regression line plot that 
        includes the regression line itself placed on 
        top of the sample data.
        """
        if self.debug:
            print(f"Debugging > Currently constructing regression line plot.")
        self.find_central_coordinates_for_each_quadrant()
        self.determine_least_dense_quadrant_in_plot()
        
            # General Plot Styles:
        plt.rc('font', family = self._font_family)
        plt.style.use(self._plot_background)
        plt.clf()
        
            # Set up the Figure:
        figure, (axes1, axes2) = plt.subplots(
            nrows = 1,
            ncols = 2,
            gridspec_kw = {
                'width_ratios': [12, 8.5]
            })

            # Prepare the Graph with Annotations and Labels:
        axes1.clear()
        axes1.grid(False)
        axes1.set_title (self.plot_title, fontsize = 12)
        axes1.set_xlabel(self.x_label,    fontsize = 13)
        axes1.set_ylabel(self.y_label,    fontsize = 13)
        axes1.yaxis.label.set_size(self.graph_axes_label_size)
        axes1.xaxis.label.set_size(self.graph_axes_label_size)
        
            # Plot the Raw Data:
        axes1.scatter(
            self.x_data, 
            self.y_data, 
            s = self._scatter_plot_dots_size, 
            c = self._scatter_plot_dots_color
        )
        
            # Preparation for the Graph:
        x_range = self.obtain_iterable_range_for_x_axis()
        regression_line_corresponding_to_x_range = self.regression_line_at_x(x_range)
        
            # Plot the Regression Line:
        axes1.plot(
            x_range, 
            regression_line_corresponding_to_x_range, 
            color     = self._regression_line_plot_color,
            linewidth = self._regression_line_thickness
        )
        
            # Annotate the Regression line with its Equation:
        axes1.annotate(
            self.format_latex_regression_line_equation(),
            (self.equation_text_coordinates[0], self.equation_text_coordinates[1]),
            horizontalalignment = "center",
            fontsize = self._regression_line_fontsize,
        )
    
            # Prepare the Table's Styling:
        cellColors = [[self._cell_colors] * 3] * 6
        columnColors = [self._column_colors] * 3
        
        axes2.axis("off")
        the_table        = axes2.table(
            cellText     = self.formatted_confidence_interval_grid_table,
            cellLoc      = "center",
            cellColours  = cellColors,
            colColours   = columnColors,
            colLabels    = ("CI", "$\\beta_{0}$","$\\beta_{1}$"),
            loc          = "best")
        the_table.auto_set_column_width(col = list(range(4)))
        the_table.set_fontsize(self._only_fitting_table_font_size)
        the_table.auto_set_font_size(False)

            # Final Preparation for the Graph:
        figure.tight_layout()
        figure.savefig(self.specified_plot_name, dpi = self._default_dots_per_inch)
        return self
    
    def helpful_stuff(self):
        print("https://stackoverflow.com/a/44407301")
        print("https://stackoverflow.com/a/3459131")
        print("https://stackoverflow.com/a/42827330")
    
    def compute_regression_line(self):
        """
        Perform in sequence all necessary computations
        to ascertain the regression line slope an
        y-intercept and construct confidence intervals
        for the slope and y-intercept given sample data
        """
        if self.debug or self.verbose:
            (f"Debugging > Commenced constructing regression line and associated plot")
        self.compute_sample_size()
        self.compute_degrees_of_freedom()
        self.compute_sample_x_mean()
        self.compute_sample_y_mean()
        self.calculate_sxx()
        self.calculate_sxy()
        self.compute_regression_line_slope()
        self.compute_regression_line_y_int()
        self.compute_residuals()
        self.compute_residuals_squared()
        self.compute_rss()
        self.sample_variance()
        self.compute_estimated_standard_error_in_regression_line_slope()
        self.compute_t_statistic_for_regression_line_slope()
        self.compute_p_value_given_test_statistic_for_slope()
        self.compute_confidence_interval_for_slope()
        self.compute_estimated_standard_error_in_regression_line_y_int()
        self.compute_t_statistic_for_regression_line_y_int()
        self.compute_p_value_given_test_statistic_for_y_int()
        self.compute_confidence_interval_for_y_int()
        self.validate_regression_line_plot_settings()
        self.get_regression_line_statistical_summary()
        return self
      
if __name__ == "__main__":
    try:
        import sys
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.integrate as integrate
        from scipy.special import gamma
    except Exception as e:
        print(f"Error importing necessary modules: {e}")
        
    array_of_possible_confirmation = ["yes", "y", "Y", "ye", "YES", "true", "TRUE"]
    array_of_possible_rejections   = ["no", "n", "N", "nop", "nope", "NO", "NOPE", "false", "FALSE", "stop", "quit"]
    
    while True:
        try:
            does_user_want_verbose_on = input("Would you like to see verbose output? > ")
            if does_user_want_verbose_on.lower() not in array_of_possible_confirmation and does_user_want_verbose_on.lower() not in array_of_possible_rejections:
                print(f"That wasn't understood as a valid input. Please try: {', '.join(array_of_possible_confirmation)} or {', '.join(array_of_possible_confirmation)}")
                continue
            if does_user_want_verbose_on.lower() in array_of_possible_confirmation:
                verbose_on_or_off = True
            if does_user_want_verbose_on.lower() in array_of_possible_rejections:
                verbose_on_or_off = False
        except:
            print(f"There was an error understanding your input. Valid responses are: {', '.join(array_of_possible_confirmation)}")
            continue
        else:
            break
    while True:
        try:
            does_user_want_to_debug = input("Would you like to see debugging output? (Hint: NO, you DON'T) > ")
            if does_user_want_to_debug.lower() not in array_of_possible_confirmation and does_user_want_to_debug.lower() not in array_of_possible_rejections:
                print(f"That wasn't understood as a valid input. Please try: {', '.join(array_of_possible_confirmation)} or {', '.join(array_of_possible_rejections)}")
                continue
            if does_user_want_to_debug.lower() in array_of_possible_confirmation:
                debugging_on_or_off = True
            if does_user_want_to_debug.lower() in array_of_possible_rejections:
                debugging_on_or_off = False
        except:
            print(f"There was an error understanding your input. Valid responses are: {', '.join(array_of_possible_confirmation)}")
            continue
        else:
            break
    while True:
        try:
            raw_x_data = input("Please enter your x-values for the raw data > ")
            run_size_x = [float(i.strip().replace(",", "")) for i in raw_x_data.split(" ")] or [float(i.strip()) for i in raw_x_data.split(",")]
        except Exception as e:
            print(f"Could not parse your x-values {e}")
        else:
            break
    while True:
        try:
            raw_y_data = input("Please enter your y-values for the raw data > ")
            run_size_y = [float(j.strip().replace(",", "")) for j in raw_y_data.split(" ")] or [float(j.strip()) for j in raw_y_data.split(",")]
        except Exception as e:
            print(f"Could not parse your y-values: {e}")
        else:
            break
    try:
        if verbose_on_or_off is True or verbose_on_or_off is False:
            if debugging_on_or_off is True or debugging_on_or_off is False:
                if run_size_x is not None and run_size_y is not None:
                    regline = RegressionLine(
                    run_size_x, 
                    run_size_y, 
                    debug = debugging_on_or_off, 
                    verbose = verbose_on_or_off)
                regline.compute_regression_line()
                regline.construct_regression_line_plot()
    except:
        sys.exit("To run this program, you must specify: (1) a list of x-data, (2) a list of y-data, (3) true or false for verbose output, and (4) true or false for debugging output.")
