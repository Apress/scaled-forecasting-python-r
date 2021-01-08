import pandas as pd
import numpy as np
import os
import pandas_datareader as pdr
from collections import Counter
from scipy.stats import pearsonr
import rpy2.robjects as ro

# make the working directory friendly for R
rwd = os.getcwd().replace('\\','/')

class Forecaster:
    """ object to forecast time series data
        natively supports the extraction of FRED data, could be expanded to other APIs with few adjustments
        the following models are supported:
            adaboost (sklearn)
            arima (R forecast pkg: auto.arima)
            arima (statsmodels)
            arima-x13 (R seasonal pkg: seas)
            average (any number of models can be averaged)
            ets (R forecast pkg: ets)
            gradient boosted trees (sklearn)
            lasso (sklearn)
            multi level perceptron (sklearn)
            multi linear regression (sklearn)
            naive (propagates final observed value forward)
            random forest (sklearn)
            ridge (sklearn)
            support vector regressor (sklearn)
            tbats (R forecast pkg: tbats)
            var (R vars pkg: VAR)      
            vecm (R tsDyn pkg: VECM)
        more models can be added by building more methods

        for every evaluated model, the following information is stored in the object attributes:
            in self.info (dict), a key is added as the model name and a nested dictionary as the value
                    the nested dictionary has the following keys:
                        'holdout_periods' : int - the number of periods held out in the test set
                        'model_form' : str - the name of the model with any hyperparameters, external regressors, etc
                        'test_set_actuals' : list - the actual figures from the test set
                        'test_set_predictions' : list - the predicted figures from the test set evaluated with a model of the training set
                        'test_set_ape' : list - the absolute percentage error for each period from the forecasted training set figures, evaluated with the actual test set figures
                in self.mape (dict), a key is added as the model name and the Mean Absolute Percent Error as the value
                in self.forecasts (dict), a key is added as the model name and a list of forecasted figures as the value
                in self.feature_importance (dict), a key is added to the dictionary as the model name and the value is a dataframe that gives some info about the features' prediction power
                    if it is an sklearn model, it will be permutation feature importance from the eli5 package (https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html)
                    any other model, it is summary output with at least the coefficient magnitudes
                    index is always the feature names (depending on how the models were run, the index won't always be labeled with the explicit variable names, but still interpretable)
                    if the model doesn't use external regressors, no key is added here

        Author Michael Keith: mikekeith52@gmail.com
    """
    def __init__(self,name=None,y=None,current_dates=None,future_dates=None,
                 current_xreg=None,future_xreg=None,forecast_out_periods=24,**kwargs):
        """ You can load the object with data using __init__ or you can leave all default arguments and load the data with an attached API method (such as get_data_fred())
            Parameters: name : str
                        y : list
                        current_dates : list
                            an ordered list of dates that correspond to the ordered values in self.y
                            dates must be a pandas datetime object (pd.to_datetime())
                        future_dates : list
                            an ordered list of dates that correspond to the future periods being forecasted
                            dates must be a pandas datetime object (pd.to_datetime())
                        current_xreg : dict
                        future_xreg : dict
                        forecast_out_periods : int, default length of future_dates or 24 if that is None
                        **all keyword arguments become attributes
        """
        self.name = name
        self.y = y
        self.current_dates = current_dates
        self.future_dates = future_dates
        self.current_xreg = current_xreg
        self.future_xreg = future_xreg
        self.forecast_out_periods=forecast_out_periods if future_dates is None else len(future_dates)
        self.info = {}
        self.mape = {}
        self.forecasts = {}
        self.feature_importance = {}
        self.ordered_xreg = None
        self.best_model = ''

        for key, value in kwargs.items():
            setattr(self,key,value)

    def _score_and_forecast(self,call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars):
        """ scores a model on a test test
            forecasts out however many periods in the new data
            writes info to self.info, self.mape, and self.forecasts (this process is described in more detail in the forecast methods)
            only works within an sklearn forecast method
            Parameters: call_me : str
                        regr : sklearn.<regression_model>
                        X : pd.core.frame.DataFrame
                        y : pd.Series
                        X_train : pd.core.frame.DataFrame
                        y_train : pd.Series
                        X_test : pd.core.frame.DataFrame
                        y_test : pd.Series
                        Xvars : list or str
        """
        regr.fit(X_train,y_train)
        pred = regr.predict(X_test)
        self.info[call_me]['test_set_actuals'] = list(y_test)
        self.info[call_me]['test_set_predictions'] = list(pred)
        self.info[call_me]['test_set_ape'] = [np.abs(yhat-y) / np.abs(y) for yhat, y in zip(pred,y_test)]
        self.mape[call_me] = np.array(self.info[call_me]['test_set_ape']).mean()
        regr.fit(X,y)
        new_data = pd.DataFrame(self.future_xreg)
        if isinstance(Xvars,list):
            new_data = new_data[Xvars]
        f = regr.predict(new_data)
        self.forecasts[call_me] = list(f)

    def _set_remaining_info(self,call_me,test_length,model_form):
        """ sets the holdout_periods and model_form values in the self.info nested dictionary, where call_me (model nickname) is the key
        """
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = model_form

    def _train_test_split(self,test_length,Xvars='all'):
        """ returns an X/y full set, training set, and testing set
            resulting y objects are pd.Series
            resulting X objects are pd.core.frame.DataFrame
            this is a non-random split and the resulting test set will be size specified in test_length
            only works within an sklearn forecast method
            Parameters: test_length : int,
                            the length of the resulting test_set
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
        """
        from sklearn.model_selection import train_test_split
        X = pd.DataFrame(self.current_xreg)
        if isinstance(Xvars,list):
            X = X[Xvars]  
        y = pd.Series(self.y)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_length,shuffle=False)
        return X, y, X_train, X_test, y_train, y_test

    def _set_feature_importance(self,X,y,regr):
        """ returns the permutation feature importances of any regression model in a pandas dataframe
            leverages eli5 package (https://pypi.org/project/eli5/)
            only works within an sklearn forecast method
            Parameters: X : pd.core.frame.DataFrame
                            X regressors used to predict the depdendent variable
                        y : pd.Series
                            y series representing the known values of the dependent variable
                        regr : sklearn estimator
                            the estimator to use to score the permutation feature importance
        """
        import eli5
        from eli5.sklearn import PermutationImportance
        perm = PermutationImportance(regr).fit(X,y)
        weights_df = eli5.explain_weights_df(perm,feature_names=X.columns.tolist()).set_index('feature')
        return weights_df

    def _prepr(self,*libs,test_length,call_me,Xvars):
        """ prepares the R environment by importing/installing libraries and writing out csv files (current/future datasets) to tmp folder
            file names: tmp_r_current.csv, tmp_r_future.csv
            libs are libs to import and/or install from R
            Parameters: call_me : str
                        Xvars : list, "all", or starts with "top_"
                        *libs: str
                            library names to import into the R environment
                            if library name not found in R environ, will attempt to install it (you will need to specify a CRAN mirror in a pop-up box)
        """
        from rpy2.robjects.packages import importr
        if isinstance(Xvars,str):
            if Xvars.startswith('top_'):
                top_xreg = int(Xvars.split('_')[1])
                if top_xreg == 0:
                    Xvars = None
                else:
                    self.set_ordered_xreg(chop_tail_periods=test_length) # have to reset here for differing test lengths in other models
                    if top_xreg > len(self.ordered_xreg):
                        Xvars = self.ordered_xreg[:]
                    else:
                        Xvars = self.ordered_xreg[:top_xreg]
            elif Xvars == 'all':
                Xvars = list(self.current_xreg.keys())
            else:
                print(f'Xvars argument not recognized: {Xvars}, changing to None')
                Xvars = None

        for lib in libs:
            try:  importr(lib)
            except: ro.r(f'install.packages("{lib}")') ; importr(lib) # install then import the r lib
        current_df = pd.DataFrame(self.current_xreg)
        current_df['y'] = self.y

        if isinstance(Xvars,list):
            current_df = current_df[['y'] + Xvars] # reorder columns 
        elif Xvars is None:
            current_df = current_df['y']
        elif Xvars != 'all':
           raise ValueError(f'unknown argument passed to Xvars: {Xvars}')

        if 'tmp' not in os.listdir():
            os.mkdir('tmp')

        current_df.to_csv(f'tmp/tmp_r_current.csv',index=False)
        
        if not Xvars is None:
            future_df = pd.DataFrame(self.future_xreg)
            future_df.to_csv(f'tmp/tmp_r_future.csv',index=False)

    def get_data_fred(self,series,date_start='1900-01-01'):
        """ imports data from FRED into a pandas dataframe
            stores the results in self.name, self.y, and self.current_dates
            Parameters: series : str
                            the name of the series to extract from FRED
                        date_start : str
                            the date to begin the time series
                            must be in YYYY-mm-dd format
            >>> f = Forecaster()
            >>> f.get_data_fred('UTUR')
        """
        self.name = series
        df = pdr.get_data_fred(series,start=date_start)
        self.y = list(df[series])
        self.current_dates = df.index.to_list()

    def process_xreg_df(self,xreg_df,date_col,process_missing_columns=False,**kwargs):
        """ takes a dataframe of external regressors
            any non-numeric data will be made into a 0/1 binary variable (using pd.get_dummies(drop_first=True))
            deals with columns with missing data
            eliminates rows that don't correspond with self.y's timeframe
            splits values between the future and current xregs
            changes self.forecast_out_periods
            assumes the dataframe is aggregated to the same timeframe as self.y (monthly, quarterly, etc.)
            for more complex processing, perform manipulations before passing through this function
            stores results in self.xreg
            Parameters: xreg_df : pandas dataframe
                        date_col : str, requried
                            the name of the date column in xreg_df, if applicable
                            if no date column available, pas None -- assumes none of the columns are dates and that all dataframe obs start at the same time as self.y
                            always better to pass a date column when available
                        process_missing_columns : str, dict, or bool
                            how to process columns with missing data
                            if str, one of 'remove','impute_mean','impute_median','impute_mode','forward_fill','backward_fill','impute_w_nearest_neighbors'
                            if dict, key is a column name and value is one of 'remove','impute_mean','impute_median','impute_mode','forward_fill','backward_fill','impute_w_nearest_neighbors'
                            if str, one method applied to all columns
                            if dict, the selected methods only apply to column names in the dictionary
                            if bool, only False supported -- False means this will be ignored, any unsupported argument raises an error
                        all keywords passed to the KNeighborsClassifier function (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
                            not relevant if not using this algorithm to impute missing values
            >>> xreg_df = pd.DataFrame({'date':['2020-01-01','2020-02-01','2020-03-01','2020-04-01']},'x1':[1,2,3,5],'x2':[1,3,3,3])
            >>> f = Forecaster(y=[4,5,9],current_dates=['2020-01-01','2020-02-01','2020-03-01'])
            >>> f.process_xreg_df(xreg_df,date_col='date')
            >>> print(f.current_xreg)
            {'x1':[1,2,3],'x2':[1,3,3]}

            >>> print(f.future_xreg)
            {'x1':[5],'x2':[3]}

            >>> print(f.future_dates)
            ['2020-04-01']

            >>> print(f.forecast_out_periods)
            1
        """
        def _remove_(c):
            xreg_df.drop(columns=c,inplace=True)
        def _impute_mean_(c):
            xreg_df[c].fillna(xreg_df[c].mean(),inplace=True)
        def _impute_median_(c):
            xreg_df[c].fillna(xreg_df[c].median(),inplace=True)
        def _impute_mode_(c):
            from scipy.stats import mode
            xreg_df[c].fillna(mode(xreg_df[c])[0][0],inplace=True)
        def _forward_fill_(c):
            xreg_df[c].fillna(method='ffill',inplace=True)
        def _backward_fill_(c):
            xreg_df[c].fillna(method='bfill',inplace=True)
        def _impute_w_nearest_neighbors_(c):
            from sklearn.neighbors import KNeighborsClassifier
            predictors=[e for e in xreg_df if len(xreg_df[e].dropna())==len(xreg_df[e])] # predictor columns can have no NAs
            predictors=[e for e in predictors if e != c] # predictor columns cannot be the same as the column to impute (this should be taken care of in the line above, but jic)
            predictors=[e for e in predictors if xreg_df[e].dtype in (np.int32,np.int64,np.float32,np.float64,int,float)] # predictor columns must be numeric -- good idea to dummify as many columns as possible
            clf = KNeighborsClassifier(**kwargs)
            df_complete = xreg_df.loc[xreg_df[c].isnull()==False]
            df_nulls = xreg_df.loc[xreg_df[c].isnull()]
            trained_model = clf.fit(df_complete[predictors],df_complete[c])
            imputed_values = trained_model.predict(df_nulls[predictors])
            df_nulls[c] = imputed_values
            xreg_df[c] = pd.concat(df_complete[c],df_nulls[c])

        if not date_col is None:
            xreg_df[date_col] = pd.to_datetime(xreg_df[date_col])
            self.future_dates = list(xreg_df.loc[xreg_df[date_col] > self.current_dates[-1],date_col])
            xreg_df = xreg_df.loc[xreg_df[date_col] >= self.current_dates[0]]
        xreg_df = pd.get_dummies(xreg_df,drop_first=True)

        if not not process_missing_columns:
            if isinstance(process_missing_columns,dict):
                for c, v in process_missing_columns.items():
                    if (v == 'remove') & (xreg_df[c].isnull().sum() > 0):
                        _remove_(c)
                    elif v == 'impute_mean':
                        _impute_mean_(c)
                    elif v == 'impute_median':
                        _impute_median_(c)
                    elif v == 'impute_mode':
                        _impute_mode_(c)
                    elif v == 'forward_fill':
                        _forward_fill_(c)
                    elif v == 'backward_fill':
                        _backward_fill_(c)
                    elif v == 'impute_w_nearest_neighbors':
                        _impute_w_nearest_neighbors_(c)
                    else:
                        raise ValueError(f'argument {v} not supported for columns {c} in process_missing')

            elif isinstance(process_missing_columns,str):
                for c in xreg_df:
                    if xreg_df[c].isnull().sum() > 0:
                        if process_missing_columns == 'remove':
                            _remove_(c)
                        elif process_missing_columns == 'impute_mean':
                            _impute_mean_(c)
                        elif process_missing_columns == 'impute_median':
                            _impute_median_(c)
                        elif process_missing_columns == 'impute_mode':
                            _impute_mode_(c)
                        elif process_missing_columns == 'forward_fill':
                            _forward_fill_(c)
                        elif process_missing_columns == 'backward_fill':
                            _backward_fill_(c)
                        elif process_missing_columns == 'impute_w_nearest_neighbors':
                            _impute_w_nearest_neighbors_(c)
                        else:
                            raise ValueError(f'argument passed to process_missing not recogized: {process_missing}')
            else:
                raise ValueError(f'argument passed to process_missing not recogized: {process_missing}')

        if not date_col is None:
            current_xreg_df = xreg_df.loc[xreg_df[date_col].isin(self.current_dates)].drop(columns=date_col)
            future_xreg_df = xreg_df.loc[xreg_df[date_col] > self.current_dates[-1]].drop(columns=date_col)        
        else:
            current_xreg_df = xreg_df.iloc[:len(self.y)]
            future_xreg_df = xreg_df.iloc[len(self.y):]

        assert current_xreg_df.shape[0] == len(self.y), 'something is wrong with the passed dataframe--make sure the dataframe is at least one observation greater in length than y and specify a date column if one available'
        self.forecast_out_periods = future_xreg_df.shape[0]
        self.current_xreg = current_xreg_df.to_dict(orient='list')
        self.future_xreg = future_xreg_df.to_dict(orient='list')

    def set_and_check_data_types(self,check_xreg=True):
        """ changes all attributes in self to the object type they should be (list, str, dict, etc.)
            if a conversion is unsuccessful, will raise an error
            Parameters: check_xreg : bool, default True
                if True, checks that self.current_xreg and self.future_xregs are dict types (raises an error if check fails)
                change this to False if wanting to perform auto-regressive forecasts only
        """
        self.name = str(self.name) if not isinstance(self.name,str) else self.name
        self.y = list(self.y) if not isinstance(self.y,list) else self.y
        self.current_dates = list(self.current_dates) if not isinstance(self.current_dates,list) else self.current_dates
        self.future_dates = list(self.future_dates) if not isinstance(self.future_dates,list) else self.future_dates
        self.forecast_out_periods = int(self.forecast_out_periods)
        if check_xreg:
            assert isinstance(self.current_xreg,dict), f'current_xreg must be dict type, not {type(self.current_xreg)}'
            assert isinstance(self.future_xreg,dict), f'future_xreg must be dict type, not {type(self.future_xreg)}'
        
    def check_xreg_future_current_consistency(self):
        """ checks that the self.y is same size as self.current_dates
            checks that self.y is same size as the values in self.current_xreg
            checks that self.future_dates is same size as the values in self.future_xreg
            if any of these checks fails, raises an AssertionError
        """
        for k, v in self.current_xreg.items():
            assert len(self.y) == len(self.current_dates), f'the length of y ({len(self.y)}) and the length of current_dates ({len(self.current_dates)}) do not match!'
            assert len(self.current_xreg[k]) == len(self.y), f'the length of {k} ({len(v)}) stored in the current_xreg dict is not the same length as y ({len(self.y)})'
            assert k in self.future_xreg.keys(), f'{k} not found in the future_xreg dict!'
            assert len(self.future_xreg[k]) == len(self.future_dates), f'the length of {k} ({len(self.future_xreg[k])}) stored in the future_xreg dict is not the same length as future_dates ({len(self.future_dates)})'

    def set_forecast_out_periods(self,n):
        """ sets the self.forecast_out_periods attribute and truncates self.future_dates and self.future_xreg if needed
            Parameters: n : int
                the number of periods you want to forecast out for
                if this is a larger value than the size of self.future_dates, some models may fail
        """
        if isinstance(n,int):
            if n >= 1:
                self.forecast_out_periods = n
                if isinstance(self.future_dates,list):
                    self.future_dates = self.future_dates[:n]
                if isinstance(self.future_xreg,dict):
                    for k,v in self.future_xreg.items():
                        self.future_xreg[k] = v[:n]
            else:
                raise ValueError(f'n must be greater than 1, got {n}')  
        else:
            raise ValueError(f'n must be an int type, got {type(n)}')

    def set_ordered_xreg(self,chop_tail_periods=0,include_only='all',exclude=None,quiet=True):
        """ method for ordering stored externals from most to least correlated, according to absolute Pearson correlation coefficient value
            will not error out if a given external has no variation in it -- will simply skip
            when measuring correlation, will log/difference variables when possible to compare stationary results
            stores the results in self.ordered_xreg as a list
            if two vars are perfectly correlated, will skip the second one
            resuting self.ordered_xreg attribute may therefore not contain all xregs but will contain as many as could be set
            Parameters: chop_tail_periods : int, default 0
                            The number of periods to chop (to compare to a training dataset)
                            This is used to reduce the chance of overfitting the data by using mismatched test periods for forecasts
                        include_only : list or any other data type, default "all"
                            if this is a list, only the externals in the list will be considered when testing correlation
                            if this is not a list, then it will be ignored and all externals will be considered
                            if this is a list, exclude will be ignored
                        exclude : list or any other data type, default None
                            if this is a list, the externals in the list will be excluded when testing correlation
                            if this is not a list, then it will be ignored and no externals will be excluded
                            if include_only is a list, this is ignored
                            note: it is possible for include_only to be its default value, "all", and exclude to not be ignored if it is passed as a list type
                        quiet : bool, default True
                            if this is True, then if a given external is ignored (either because no correlation could be calculated or there are no observations after its tail has been chopped), you will not know
                            if this is False, then if a given external is ignored, it will print which external is being skipped
            >>> f = Forecaster()
            >>> f.get_data_fred('UTUR')
            >>> f.process_xreg_df(xreg_df,chop_tail_periods=12)
            >>> f.set_ordered_xreg()
            >>> print(f.ordered_xreg)
            [x2,x1] # in this case x2 correlates more strongly than x1 with y on a test set with 12 holdout periods
        """
        def log_diff(x):
            """ returns the logged difference of an array
            """
            return np.diff(np.log(x),n=1)

        if isinstance(include_only,list):
            use_these_externals = {}
            for e in include_only:
                use_these_externals[e] = self.current_xreg[e]
        else:
            use_these_externals = self.current_xreg.copy()
            if isinstance(exclude,list):
                for e in exclude:
                    use_these_externals.pop(e)
                
        ext_reg = {}
        for k, v in use_these_externals.items():
            if chop_tail_periods > 0:
                x = np.array(v[:(chop_tail_periods*-1)])
                y = np.array(self.y[:(chop_tail_periods*-1)])
            else:
                x = np.array(v)
                y = np.array(self.y[:])
                
            if (x.min() <= 0) & (y.min() > 0):
                y = log_diff(y)
                x = x[1:]
            elif (x.min() > 0) & (y.min() > 0):
                y = log_diff(y)
                x = log_diff(x)
            
            if len(np.unique(x)) == 1:
                if not quiet:
                    print(f'no variation in {k} for time period specified')
                continue
            else: 
                r_coeff = pearsonr(y,x)
            
            if np.abs(r_coeff[0]) not in ext_reg.values():
                ext_reg[k] = np.abs(r_coeff[0])
        
        k = Counter(ext_reg) 
        self.ordered_xreg = [h[0] for h in k.most_common()] # this should give us the ranked external regressors

    def forecast_auto_arima(self,test_length=1,Xvars=None,call_me='auto_arima'):
        """ Auto-Regressive Integrated Moving Average 
            forecasts using auto.arima from the forecast package in R
            uses an algorithm to find the best ARIMA model automatically by minimizing in-sample aic, checks for seasonality (but doesn't work very well)
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list, "all", None, or starts with "top_", default None
                            the independent variables used to make predictions
                            if it is a list, will attempt to estimate a model with that list of Xvars
                            if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars
                            "top" is determined through absolute value of the pearson correlation coefficient on the training set
                            if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors that are not perfectly colinear and have variation
                            if it is "all", will attempt to estimate a model with all available x regressors, regardless of whether there is collinearity or no variation
                            because the auto.arima function fails in the cases of perfect collinearity or no variation, using "top_" or a list with one element is safest option
                            if no arima model can be estimated, will raise an error
                        call_me : str, default "auto_arima"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            >>> f = Forecaster()
            >>> f.get_data_fred('UTUR')
            >>> f.forecast_auto_arima(test_length=12,call_me='arima')
            >>> print(f.info['arima'])
            {'holdout_periods': 12, 
            'model_form': 'ARIMA(0,1,5)',
            'test_set_actuals': [2.4, 2.4, ..., 5.0, 4.1],
            'test_set_predictions': [2.36083282553252, 2.3119957980461803, ..., 2.09177057271149, 2.08127132827637], 
            'test_set_ape': [0.0163196560281154, 0.03666841748076, ..., 0.581645885457702, 0.49237284676186205]}

            >>> print(f.forecasts['arima'])
            [4.000616524942799, 4.01916650578768, ..., 3.7576542462753904, 3.7576542462753904]
            
            >>> print(f.mape['arima'])
            0.4082393522799069

            >>> print(f.feature_importance['arima'])
                coef        se    tvalue          pval
            ma5  0.189706  0.045527  4.166858  3.598788e-05
            ma4 -0.032062  0.043873 -0.730781  4.652316e-01
            ma3 -0.060743  0.048104 -1.262753  2.072261e-01
            ma2 -0.257684  0.044522 -5.787802  1.213441e-08
            ma1  0.222933  0.042513  5.243861  2.265347e-07
        """
        from scipy import stats
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        self._prepr('forecast',test_length=test_length,call_me=call_me,Xvars=Xvars)
        ro.r(f"""
            rm(list=ls())
            setwd('{rwd}')
            data <- data.frame(read.csv('tmp/tmp_r_current.csv'))
            data_train <- data[1:(nrow(data)-{test_length}),,drop=FALSE]
            data_test <- data[(nrow(data)-{test_length} + 1):nrow(data),,drop=FALSE]
            
            y <- data$y
            y_train <- y[1:(nrow(data)-{test_length})]
            y_test <- y[(nrow(data)-{test_length} + 1):nrow(data)]
            
            """)

        ro.r("""
            if (ncol(data) > 1){
                future_externals = read.csv('tmp/tmp_r_future.csv')
                externals = names(data)[2:ncol(data)]
                xreg_c <- as.matrix(data[,externals])
                xreg_tr <- as.matrix(data_train[,externals])
                xreg_te <- as.matrix(data_test[,externals])
                xreg_f <- as.matrix(future_externals[,externals])
            } else {
                xreg_c <- NULL
                xreg_tr <- NULL
                xreg_te <- NULL
                xreg_f <- NULL
            }
            ar <- auto.arima(y_train,xreg=xreg_tr)
            f <- forecast(ar,xreg=xreg_te,h=length(y_test))
            # f[[4]] are point estimates, f[[1]] is the ARIMA form
            p <- f[[4]]
            arima_form <- f[[1]]
            write <- data.frame(actual=y_test,
                                forecast=p)
            write$APE <- abs(write$actual - write$forecast) / abs(write$actual)
            write$model_form <- arima_form
            write.csv(write,'tmp/tmp_test_results.csv',row.names=F)
        """)
        
        ro.r(f"""
            ar <- auto.arima(y,max.order=10,stepwise=F,xreg=xreg_c)
            f <- forecast(ar,xreg=xreg_f,h={self.forecast_out_periods})
            p <- f[[4]]
            arima_form <- f[[1]]
            
            write <- data.frame(forecast=p)
            write$model_form <- arima_form
            write.csv(write,'tmp/tmp_forecast.csv',row.names=F)
        """)
        
        ro.r("""
            summary_df = data.frame(coef=rev(coef(ar)),se=rev(sqrt(diag(vcov(ar)))))
            if (exists('externals')){row.names(summary_df)[1:length(externals)] <- externals}
            summary_df$tvalue = summary_df$coef/summary_df$se
            write.csv(summary_df,'tmp/tmp_summary_output.csv')
        """)

        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        self.mape[call_me] = tmp_test_results['APE'].mean()
        self.forecasts[call_me] = list(tmp_forecast['forecast'])
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = list(tmp_test_results['actual'])
        self.info[call_me]['test_set_predictions'] = list(tmp_test_results['forecast'])
        self.info[call_me]['test_set_ape'] = list(tmp_test_results['APE'])
        self.feature_importance[call_me] = pd.read_csv('tmp/tmp_summary_output.csv',index_col=0)

        if self.feature_importance[call_me].shape[0] > 0: # for the (0,i,0) model case
            self.feature_importance[call_me]['pval'] = stats.t.sf(np.abs(self.feature_importance[call_me]['tvalue']), len(self.y)-1)*2 # https://stackoverflow.com/questions/17559897/python-p-value-from-t-statistic
        else:
            self.feature_importance.pop(call_me)

    def forecast_sarimax13(self,start='auto',interval=12,test_length=1,Xvars=None,call_me='sarimax13',X13_PATH='auto',error='raise'):
        """ Seasonal Auto-Regressive Integrated Moving Average - ARIMA-X13 - https://www.census.gov/srd/www/x13as/
            Forecasts using the seas function from the seasonal package, also need the X13 software (x13as.exe) saved locally
            Automatically takes the best model ARIMA model form that fulfills a certain set of criteria (low forecast error rate, high statistical significance, etc)
            X13 is a sophisticated way to model seasonality with ARIMA maintained by the census bureau, and the seasonal package provides a simple wrapper around the software with R
            The function here is simplified, but the power in X13 is its database offers precise ways to model seasonality, also takes into account outliers
            Documentation: https://cran.r-project.org/web/packages/seasonal/seasonal.pdf, http://www.seasonal.website/examples.html
            This package only allows for monthly or less granular observations, and only three years or fewer of predictions
            when a series does not have a lot of seasonality, sometimes the model fails, also when it tries to add the same outlier in two different ways
            Parameters: start : tuple of length 2 or "auto", default "auto"
                            1st element is the start year
                            2nd element is the start period in the appropriate interval
                            for instance, if you have quarterly data and your first obs is 2nd quarter of 1980, this would be (1980,2)
                            if "auto", assumes the dates in self.current_dates are monthly in yyyy-mm-01 format and will use the first element in the list 
                        interval : 1 of {1,2,4,12}, default 12
                            1 for annual, 2 for bi-annual, 4 for quarterly, 12 for monthly
                            unfortunately, x13 does not allow for more granularity than the monthly level
                        test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list, "all", None, or starts with "top_", default None
                            the independent variables used to make predictions
                            if it is a list, will attempt to estimate a model with that list of Xvars
                            if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars
                            "top" is determined through absolute value of the pearson correlation coefficient on the training set
                            if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors that are not perfectly colinear and have variation
                            if it is "all", will attempt to estimate a model with all available x regressors, regardless of whether there is collinearity or no variation
                            because the seas function fails in the cases of perfect collinearity or no variation, using "top_" or a list with one element is safest option
                            x13 already has an extensive list of x regressors that it will pull automatically--read the documentation for more info
                        call_me : str, default "sarimax13"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        X13_PATH : str, default "auto"
                            the local location of the x13as.exe executable
                            if "auto", assumes it is saved in {local_directory}/x13asall_V1.1_B39/x13as
                            be sure to use front slashes (/) and not backslashes (\\) to keep R happy
                        error: one of {"raise","pass","print"}, default "raise"
                            if unable to estimate the model, "raise" will raise an error
                            if unable to estimate the model, "pass" will silently skip the model and delete all associated attribute keys (self.info)
                            if unable to estimate the model, "print" will skip the model, delete all associated attribute keys (self.info), and print the error
                            errors are common even if you specify everything correctly -- it has to do with the X13 estimator itself
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        if start == 'auto':
            try: start = tuple(np.array(str(self.current_dates[0]).split('-')[:2]).astype(int))
            except: raise ValueError('could not set start automatically, try passing argument manually')
        if X13_PATH == 'auto':
            X13_PATH = f'{rwd}/x13asall_V1.1_B39/x13as'

        assert os.path.exists(X13_PATH + '/x13as.exe'), 'x13as.exe not found. did you specify X13_PATH?'
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        self._prepr('forecast','seasonal',test_length=test_length,call_me=call_me,Xvars=Xvars)

        ro.r(f"""
            rm(list=ls())
            Sys.setenv(X13_PATH = '{X13_PATH}')
            setwd('{rwd}')
            start_p <- c{start}
            interval <- {interval}
            test_length <- {test_length}
            
            data <- data.frame(read.csv('tmp/tmp_r_current.csv'))
            
            y <- ts(data$y,start=start_p,deltat=1/interval)
            y_train <- subset(y,start=1,end=nrow(data)-test_length)
            y_test <- subset(y,start=nrow(data)-test_length+1,end=nrow(data))
            
            """)

        ro.r("""
            if (ncol(data) > 1){
              future_externals = data.frame(read.csv('tmp/tmp_r_future.csv'))
              r <- max(0,36-nrow(future_externals))
              filler <-as.data.frame(replicate(ncol(future_externals),rep(0,r))) # we always need at least three years of data for this package
              # if we have less than three years, fill in the rest with 0s
              # we still only use predictions matching whatever is stored in self.forecast_out_periods
              # https://github.com/christophsax/seasonal/issues/200
              names(filler) <- names(future_externals)
              future_externals <- rbind(future_externals,filler)
              externals <- names(data)[2:ncol(data)]
              data_c <- data[,externals, drop=FALSE]
              data_f <- future_externals[,externals, drop=FALSE]
              all_externals_ts <- ts(rbind(data_c,data_f),start=start_p,deltat=1/interval)
            } else {
              all_externals_ts <- NULL
            }
        """)

        try:
            ro.r(f"""
                    m_test <- seas(x=y_train,xreg=all_externals_ts,forecast.save="forecasts",pickmdl.method="best")
                    p <- series(m_test, "forecast.forecasts")[1:test_length,]
                    m <- seas(x=y,xreg=all_externals_ts,forecast.save="forecasts",pickmdl.method="best")
                    f <- series(m, "forecast.forecasts")[1:{self.forecast_out_periods},]
                    arima_form <- paste('ARIMA-X13',m_test$model$arima$model)
                    write <- data.frame(actual=y_test,forecast=p[,1])
                    write$APE <- abs(write$actual - write$forecast) / abs(write$actual)
                    write$model_form <- arima_form
                    write.csv(write,'tmp/tmp_test_results.csv',row.names=F)
                    arima_form <- paste('ARIMA-X13',m$model$arima$model)
                    write <- data.frame(forecast=f[,1])
                    write$model_form <- arima_form
                    write.csv(write,'tmp/tmp_forecast.csv',row.names=F)
            """)
        except Exception as e:
            self.info.pop(call_me)
            if error == 'raise':
                raise e
            else:
                if error == 'print':
                    print(f"skipping model, here's the error:\n{e}")
                elif error != 'pass':
                    print(f'error argument not recognized: {error}')
                    raise e
                return None

        ro.r("""
            # feature_importance -- cool output
            summary_df <- data.frame(summary(m))
            if (exists("externals")) {summary_df$term[1:length(externals)] <- externals}
            write.csv(summary_df,'tmp/tmp_summary_output.csv',row.names=F)
        """)

        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        self.mape[call_me] = tmp_test_results['APE'].mean()
        self.forecasts[call_me] = list(tmp_forecast['forecast'])
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = list(tmp_test_results['actual'])
        self.info[call_me]['test_set_predictions'] = list(tmp_test_results['forecast'])
        self.info[call_me]['test_set_ape'] = list(tmp_test_results['APE'])
        self.feature_importance[call_me] = pd.read_csv('tmp/tmp_summary_output.csv',index_col=0)

    def forecast_arima(self,test_length=1,Xvars=None,order=(0,0,0),seasonal_order=(0,0,0,0),trend=None,call_me='arima',**kwargs):
        """ ARIMA model from statsmodels: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html
            the args endog, exog, and dates passed automatically
            the args order, seasonal_order, and trend should be specified in the method
            all other arguments in the ARIMA() function can be passed to kwargs
            using this framework, the following model types can be specified:
                AR, MA, ARMA, ARIMA, SARIMA, regression with ARIMA errors
            this is meant for manual arima modeling; for a more automated implementation, see the forecast_auto_arima() and forecast_sarimax13() methods
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list, "all", or None default None
                            the independent variables to use in the resulting X dataframes
                            "top_" not supported
                        call_me : str, default "arima"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        Info about all other arguments (order, seasonal_order, trend) can be found in the sm.tsa.arima.model.ARIMA documentation (linked above)
                        other arguments from ARIMA() function can be passed as keywords
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from statsmodels.tsa.arima.model import ARIMA

        def summary_to_df(sm_model):
            """ https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe/52976810
            """
            results_summary = sm_model.summary()
            results_as_html = results_summary.tables[1].as_html()
            return pd.read_html(results_as_html, header=0, index_col=0)[0]

        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        if not Xvars is None:
            if Xvars == 'all':
                X = pd.DataFrame(self.current_xreg)
                X_f = pd.DataFrame(self.future_xreg)
            elif isinstance(Xvars,list):
                X = pd.DataFrame(self.current_xreg).loc[:,Xvars]
                X_f = pd.DataFrame(self.future_xreg).loc[:,Xvars]
            else:
                raise ValueError(f'Xvars argument not recognized: {Xvars}')
        else:
            X = None
            X_f = None

        y = pd.Series(self.y)

        X_train = None if Xvars is None else X.iloc[:-test_length]
        X_test = None if Xvars is None else X.iloc[-test_length:]
        y_train = y.values[:-test_length]
        y_test = y.values[-test_length:]
        dates = pd.to_datetime(self.current_dates) if not self.current_dates is None else None

        arima_train = ARIMA(y_train,exog=X_train,order=order,seasonal_order=seasonal_order,trend=trend,dates=dates,**kwargs).fit()
        pred = list(arima_train.predict(exog=X_test,start=len(y_train),end=len(y)-1,typ='levels'))
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = 'ARIMA {}x{} include {}'.format(order,seasonal_order,trend)
        self.info[call_me]['test_set_actuals'] = list(y_test)
        self.info[call_me]['test_set_predictions'] = pred
        self.info[call_me]['test_set_ape'] = [np.abs(yhat-y) / np.abs(y) for yhat, y in zip(pred,y_test)]
        self.mape[call_me] = np.array(self.info[call_me]['test_set_ape']).mean()

        arima = ARIMA(y,exog=X,order=order,seasonal_order=seasonal_order,trend=trend,dates=dates,**kwargs).fit()
        self.forecasts[call_me] = list(arima.predict(exog=X_f,start=len(y),end=len(y) + self.forecast_out_periods-1,typ='levels'))
        self.feature_importance[call_me] = summary_to_df(arima)

    def forecast_tbats(self,test_length=1,season='NULL',call_me='tbats'):
        """ Exponential Smoothing State Space Model With Box-Cox Transformation, ARMA Errors, Trend And Seasonal Component
            forecasts using tbats from the forecast package in R
            auto-regressive only (no external regressors)
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        season : int or "NULL"
                            the number of seasonal periods to consider (12 for monthly, etc.)
                            if no seasonality desired, leave "NULL" as this will be passed directly to the tbats function in r
                        call_me : str, default "tbats"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'

        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        self._prepr('forecast',test_length=test_length,call_me=call_me,Xvars=None)
        ro.r(f"""
            rm(list=ls())
            setwd('{rwd}')
            data <- read.csv('tmp/tmp_r_current.csv')

            y <- data$y
            y_train <- y[1:(nrow(data)-{test_length})]
            y_test <- y[(nrow(data)-{test_length} + 1):nrow(data)]
        
            ar <- tbats(y_train,seasonal.periods={season})
            f <- forecast(ar,xreg=xreg_te,h=length(y_test))
            # f[[2]] are point estimates, f[[9]] is the TBATS form
            p <- f[[2]]
            tbats_form <- f[[9]]
            write <- data.frame(actual=y_test,
                                forecast=p)
            write$APE <- abs(write$actual - write$forecast) / abs(write$actual)
            write$model_form <- tbats_form
            write.csv(write,'tmp/tmp_test_results.csv',row.names=F)

            ar <- tbats(y)
            f <- forecast(ar,xreg=xreg_f,h={self.forecast_out_periods})
            p <- f[[2]]
            tbats_form <- f[[9]]
            
            write <- data.frame(forecast=p)
            write$model_form <- tbats_form
            write.csv(write,'tmp/tmp_forecast.csv',row.names=F)
        """)
        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        self.mape[call_me] = tmp_test_results['APE'].mean()
        self.forecasts[call_me] = list(tmp_forecast['forecast'])
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = list(tmp_test_results['actual'])
        self.info[call_me]['test_set_predictions'] = list(tmp_test_results['forecast'])
        self.info[call_me]['test_set_ape'] = list(tmp_test_results['APE'])

    def forecast_ets(self,test_length=1,call_me='ets'):
        """ Exponential Smoothing State Space Model
            forecasts using ets from the forecast package in R
            auto-regressive only (no external regressors)
            Parameters: test_length : int, default 1
                            the number of periods to holdout in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        call_me : str, default "ets"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'

        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        self._prepr('forecast',test_length=test_length,call_me=call_me,Xvars=None)
        ro.r(f"""
            rm(list=ls())
            setwd('{rwd}')
            data <- read.csv('tmp/tmp_r_current.csv')

            y <- data$y
            y_train <- y[1:(nrow(data)-{test_length})]
            y_test <- y[(nrow(data)-{test_length} + 1):nrow(data)]
        
            ar <- ets(y_train)
            f <- forecast(ar,xreg=xreg_te,h=length(y_test))
            # f[[2]] are point estimates, f[[8]] is the ETS form
            p <- f[[2]]
            ets_form <- f[[8]]
            write <- data.frame(actual=y_test,
                                forecast=p)
            write$APE <- abs(write$actual - write$forecast) / abs(write$actual)
            write$model_form <- ets_form
            write.csv(write,'tmp/tmp_test_results.csv',row.names=F)

            ar <- ets(y)
            f <- forecast(ar,xreg=xreg_f,h={self.forecast_out_periods})
            p <- f[[2]]
            ets_form <- f[[8]]
            
            write <- data.frame(forecast=p)
            write$model_form <- ets_form
            write.csv(write,'tmp/tmp_forecast.csv',row.names=F)
        """)
        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')
        self.mape[call_me] = tmp_test_results['APE'].mean()
        self.forecasts[call_me] = list(tmp_forecast['forecast'])
        
        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['model_form'] = tmp_forecast['model_form'][0]
        self.info[call_me]['test_set_actuals'] = list(tmp_test_results['actual'])
        self.info[call_me]['test_set_predictions'] = list(tmp_test_results['forecast'])
        self.info[call_me]['test_set_ape'] = list(tmp_test_results['APE'])

    def forecast_var(self,*series,auto_resize=False,test_length=1,Xvars=None,lag_ic='AIC',optimizer='AIC',season='NULL',max_externals=None,call_me='var'):
        """ Vector Auto Regression
            forecasts using VAR from the vars package in R
            Optimizes the final model with different time trends, constants, and x variables by minimizing the AIC or BIC in the training set
            Unfortunately, only supports a level forecast, so to avoid stationarity issues, perform your own transformations before loading the data
            Parameters: *series : required
                            lists of other series to run the VAR with
                            each list must be the same size as self.y if auto_resize is False
                            be sure to exclude NAs
                        auto_resize : bool, default False
                            if True, if series in *series are different size than self.y, all series will be truncated to match the shortest series
                            if True, note that the forecast will not necessarily make predictions based on the entire history available in y
                            using this assumes that the shortest series ends at the same time the others do and there are no periods missing
                        test_length : int, default 1
                            the number of periods to hold out in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list, "all", None, or starts with "top_", default None
                            the independent variables used to make predictions
                            if it is a list, will attempt to estimate a model with that list of Xvars
                            if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars
                            "top" is determined through absolute value of the pearson correlation coefficient on the training set
                            if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors
                            if it is "all", will attempt to estimate a model with all available x regressors
                            because the VAR function will fail if there is perfect collinearity in any of the xregs or if there is no variation in any of the xregs, using "top_" is safest option
                        lag_ic : str, one of {"AIC", "HQ", "SC", "FPE"}; default "AIC"
                            the information criteria used to determine the optimal number of lags in the VAR function
                        optimizer : str, one of {"AIC","BIC"}; default "AIC"
                            the information criteria used to select the best model in the optimization grid
                            a good, short resource to understand the difference: https://www.methodology.psu.edu/resources/AIC-vs-BIC/
                        season : int, default "NULL"
                            the number of periods to add a seasonal component to the model
                            if "NULL", no seasonal component will be added
                            don't use None ("NULL" is passed directly to the R CRAN mirror)
                            example: if your data is monthly and you suspect seasonality, you would make this 12
                        max_externals: int or None type, default None
                            the maximum number of externals to try in each model iteration
                            0 to this value of externals will be attempted and every combination of externals will be tried
                            None signifies that all combinations will be tried
                            reducing this from None can speed up processing and reduce overfitting
                        call_me : str, default "var"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        if len(series) == 0:
            raise ValueError('cannot run var -- need at least 1 series passed to *series')
        series_df = pd.DataFrame()
        min_size = min([len(s) for s in series] + [len(self.y)])
        for i, s in enumerate(series):
            if not isinstance(s,list):
                raise TypeError(f'cannot run var -- not a list type ({type(s)}) passed to *series')
            elif (len(s) != len(self.y)) & (not auto_resize):
                raise ValueError('cannot run var -- at least 1 list passed to *series is different length than y--try changing auto_resize to True')
            elif auto_resize:
                s = s[(len(s) - min_size):]
            elif not not auto_resize:
                raise ValueError(f'argument in auto_resize not recognized: {auto_resize}')
            series_df[f'cid{i+1}'] = s

        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        assert optimizer in ('AIC','BIC'), f'cannot estimate model - optimizer value of {optimizer} not recognized'
        assert lag_ic in ("AIC", "HQ", "SC", "FPE"), f'cannot estimate model - lag_ic value of {lag_ic} not recognized'

        if (max_externals is None) & (not Xvars is None):
            if isinstance(Xvars,list):
                max_externals = len(Xvars)
            elif Xvars == 'all':
                max_externals = len(self.current_xreg.keys())
            elif Xvars.startswith('top_'):
                max_externals = int(Xvars.split('_')[1])
            else:
                raise ValueError(f'Xvars argument {Xvars} not recognized')

        self._prepr('vars',test_length=test_length,call_me=call_me,Xvars=Xvars)
        series_df.to_csv('tmp/tmp_r_cid.csv',index=False)
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])

        ro.r(f"""
                rm(list=ls())
                setwd('{rwd}')
                data <- data.frame(read.csv('tmp/tmp_r_current.csv'))
                cid <- na.omit(data.frame(read.csv('tmp/tmp_r_cid.csv')))
                test_periods <- {test_length}
                max_ext <- {max_externals if not max_externals is None else 0}
                lag_ic <- "{lag_ic}"
                IC <- {optimizer}
                season <- {season}
                n_ahead <- {self.forecast_out_periods}
            """)

        ro.r("""
                total_length <- min(nrow(data),nrow(cid))
                data <- data[(nrow(data) - total_length + 1) : nrow(data),,drop=FALSE]
                cid <- cid[(nrow(cid) - total_length + 1) : nrow(cid),,drop=FALSE]

                if (ncol(data) > 1){
                    exogen_names <- names(data)[2:ncol(data)]
                    exogen_future <- read.csv('tmp/tmp_r_future.csv')
                    exogen_train <- data[1:(nrow(data)-test_periods),names(data)[2:ncol(data)], drop=FALSE]
                    exogen_test <- data[(nrow(data)-(test_periods-1)):nrow(data),names(data)[2:ncol(data)], drop=FALSE]

                    # every combination of the external regressors, including no external regressors
                    exogenstg1 <- list()
                    for (i in 1:length(exogen_names)){
                      exogenstg1[[i]] <- combn(exogen_names,i)
                    }

                    h <- 2
                    exogen <- list(NULL)
                    for (i in 1:min(max_ext,length(exogenstg1))) {
                      for (j in 1:ncol(exogenstg1[[i]])) {
                        exogen[[h]] <- exogenstg1[[i]][,j]
                        h <- h+1
                      }
                    }
                } else {
                    exogen <- list(NULL)
                    exogen_future <- NULL
                }
                
                data.ts <- cbind(data[[1]],cid)
                data.ts_train <- data.ts[1:(nrow(data)-test_periods),,drop=FALSE]
                data.ts_test <- data.ts[(nrow(data)-(test_periods-1)):nrow(data),,drop=FALSE]

                # create a grid of parameters for the best estimator for each series pair
                include = c('none','const','trend','both')

                grid <- expand.grid(include = include, exogen=exogen)
                grid$ic <- 999999

                for (i in 1:nrow(grid)){
                  if (is.null(grid[i,'exogen'][[1]])){
                    ex_train = NULL
                  } else {
                    ex_train = exogen_train[,grid[i,'exogen'][[1]]]
                  }

                  vc_train <- VAR(data.ts_train,
                                      season=season,
                                      ic='AIC',
                                      type=as.character(grid[i,'include']),
                                      exogen=ex_train)
                  grid[i,'ic'] <-  IC(vc_train)
                }

                # choose parameters with best IC
                best_params <- grid[grid$ic == min(grid$ic),]

                # set externals
                if (is.null(best_params[1,'exogen'][[1]])){
                  ex_current = NULL
                  ex_future = NULL
                  ex_train = NULL
                  ex_test = NULL

                } else {
                  ex_current = as.matrix(data[,best_params[1,'exogen'][[1]]])
                  ex_future = as.matrix(exogen_future[,best_params[1,'exogen'][[1]]])
                  ex_train = as.matrix(exogen_train[,best_params[1,'exogen'][[1]]])
                  ex_test = as.matrix(exogen_test[,best_params[1,'exogen'][[1]]])
                }
                
                # predict on test set one more time with best parameters for model accuracy info
                vc_train <- VAR(data.ts_train,
                                 season=season,
                                 ic='AIC',
                                 type = as.character(best_params[1,'include']),
                                 exogen=ex_train)
                pred <- predict(vc_train,n.ahead=test_periods,dumvar=ex_test)
                p <- data.frame(row.names=1:nrow(data.ts_test))
                for (i in 1:length(pred$fcst)) {
                  p$col <- pred$fcst[[i]][,1]
                  names(p)[i] <- paste0('series',i)
                }
                p$xreg <- as.character(best_params[1,'exogen'])[[1]]
                p$model_form <- paste0('VAR',' include: ',best_params[1,'include'],'|selected regressors: ',as.character(best_params[1,'exogen'])[[1]])

                write.csv(p,'tmp/tmp_test_results.csv',row.names=F)

                # train the final model on full dataset with best parameter values
                vc.out = VAR(data.ts,
                              season=season,
                              ic=lag_ic,
                              type = as.character(best_params[1,'include']),
                              exogen=ex_current
                )
                # make the forecast
                fcst <- predict(vc.out,n.ahead=n_ahead,dumvar=ex_future)
                f <- data.frame(row.names=1:n_ahead)
                for (i in 1:length(fcst$fcst)) {
                  f$col <- fcst$fcst[[i]][,1]
                  names(f)[i] <- paste0('series',i)
                }
                # write final forecast values
                write.csv(f,'tmp/tmp_forecast.csv',row.names=F)

                summary_df <- coef(vc_train)[[1]]
                write.csv(summary_df,'tmp/tmp_summary_output.csv')
        """)
        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')

        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['test_set_predictions'] = list(tmp_test_results.iloc[:,0])
        self.info[call_me]['test_set_actuals'] = self.y[(-test_length):]
        self.info[call_me]['test_set_ape'] = [np.abs(y - yhat) / np.abs(y) for y, yhat in zip(self.y[(-test_length):],tmp_test_results.iloc[:,0])]
        self.info[call_me]['model_form'] = tmp_test_results['model_form'][0]
        self.mape[call_me] = np.array(self.info[call_me]['test_set_ape']).mean()
        self.forecasts[call_me] = list(tmp_forecast.iloc[:,0])
        self.feature_importance[call_me] = pd.read_csv('tmp/tmp_summary_output.csv',index_col=0)

    def forecast_vecm(self,*cids,auto_resize=False,test_length=1,Xvars=None,r=1,max_lags=6,optimizer='AIC',max_externals=None,call_me='vecm'):
        """ Vector Error Correction Model
            forecasts using VECM from the tsDyn package in R
            Optimizes the final model with different lags, time trends, constants, and x variables by minimizing the AIC or BIC in the training set
            Parameters: *cids : required
                            lists of cointegrated data
                            each list must be the same size as self.y
                            if this is only 1 list, it must be cointegrated with self.y
                            if more than 1 list, there must be at least 1 cointegrated pair between cids* and self.y (to fulfill the requirements of VECM)
                            be sure to exclude NAs
                        auto_resize : bool, default False
                            if True, if series in *series are different size than self.y, all series will be truncated to match the shortest series
                            if True, note that the forecast will not necessarily make predictions based on the entire history available in y
                            using this assumes that the shortest series ends at the same time the others do and there are no periods missing
                        test_length : int, default 1
                            the number of periods to hold out in order to test the model
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list, "all", None, or starts with "top_", default None
                            the independent variables used to make predictions
                            if it is a list, will attempt to estimate a model with that list of Xvars
                            if it begins with "top_", the character(s) after should be an int and will attempt to estimate a model with the top however many Xvars
                            "top" is determined through absolute value of the pearson correlation coefficient on the training set
                            if using "top_" and the integer is a greater number than the available x regressors, the model will be estimated with all available x regressors
                            if it is "all", will attempt to estimate a model with all available x regressors
                            because the VECM function will fail if there is perfect collinearity in any of the xregs or if there is no variation in any of the xregs, using "top_" is safest option
                        r : int, default 1
                            the number of total cointegrated relationships between self.y and *cids
                            if not an int or less than 1, an AssertionError is raised
                        max_lags : int, default 6
                            the total number of lags that will be used in the optimization process
                            1 to this number will be attempted
                            if not an int or less than 0, an AssertionError is raised
                        optimizer : str, one of {"AIC","BIC"}; default "AIC"
                            the information criteria used to select the best model in the optimization grid
                            a good, short resource to understand the difference: https://www.methodology.psu.edu/resources/AIC-vs-BIC/
                        max_externals: int or None type, default None
                            the maximum number of externals to try in each model iteration
                            0 to this value of externals will be attempted and every combination of externals will be tried
                            None signifies that all combinations will be tried
                            reducing this from None can speed up processing and reduce overfitting
                        call_me : str, default "vecm"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        if len(cids) == 0:
            raise ValueError('cannot run vecm -- need at least 1 cointegrated series in a list that is same length as y passed to *cids--no list found')
        cid_df = pd.DataFrame()
        min_size = min([len(cid) for cid in cids] + [len(self.y)])
        for i, cid in enumerate(cids):
            if not isinstance(cid,list):
                raise TypeError('cannot run var -- need at least 1 series in a list passed to *cids--not a list type detected')
            elif (len(cid) != len(self.y)) & (not auto_resize):
                raise ValueError('cannot run var -- need at least 1 series in a list that is same length as y passed to *cids--at least 1 list is different length than y--try changing auto_resize to True')
            elif auto_resize:
                cid = cid[(len(cid) - min_size):]
            elif not not auto_resize:
                raise ValueError(f'argument in auto_resize not recognized: {auto_resize}')
            cid_df[f'cid{i+1}'] = cid

        assert isinstance(r,int), f'r must be an int, not {type(r)}'
        assert r >= 1, 'r must be at least 1'
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        assert isinstance(max_lags,int), f'max_lags must be an int, not {type(max_lags)}'
        assert max_lags >= 0, 'max_lags must be positive'
        assert optimizer in ('AIC','BIC'), f'cannot estimate model - optimizer value of {optimizer} not recognized'

        if (max_externals is None) & (not Xvars is None):
            if isinstance(Xvars,list):
                max_externals = len(Xvars)
            elif Xvars == 'all':
                max_externals = len(self.current_xreg.keys())
            elif Xvars.startswith('top_'):
                max_externals = int(Xvars.split('_')[1])
            else:
                raise ValueError(f'Xvars argument {Xvars} not recognized')

        self._prepr('tsDyn',test_length=test_length,call_me=call_me,Xvars=Xvars)
        cid_df.to_csv('tmp/tmp_r_cid.csv',index=False)
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])

        ro.r(f"""
                rm(list=ls())
                setwd('{rwd}')
                data <- data.frame(read.csv('tmp/tmp_r_current.csv'))
                cid <- data.frame(read.csv('tmp/tmp_r_cid.csv'))
                test_periods <- {test_length}
                r <- {r}
                IC <- {optimizer}
                max_ext <- {max_externals if not max_externals is None else 0}
                max_lags <- {max_lags}
                n_ahead <- {self.forecast_out_periods}
            """)

        ro.r("""
                total_length <- min(nrow(data),nrow(cid))
                data <- data[(nrow(data) - total_length + 1) : nrow(data),,drop=FALSE]
                cid <- cid[(nrow(cid) - total_length + 1) : nrow(cid),,drop=FALSE]

                if (ncol(data) > 1){
                    exogen_names <- names(data)[2:ncol(data)]
                    exogen_future <- read.csv('tmp/tmp_r_future.csv')
                    exogen_train <- data[1:(nrow(data)-test_periods),names(data)[2:ncol(data)], drop=FALSE]
                    exogen_test <- data[(nrow(data)-(test_periods-1)):nrow(data),names(data)[2:ncol(data)], drop=FALSE]

                    # every combination of the external regressors, including no external regressors
                    exogenstg1 <- list()
                    for (i in 1:length(exogen_names)){
                      exogenstg1[[i]] <- combn(exogen_names,i)
                    }

                    h <- 2
                    exogen <- list(NULL)
                    for (i in 1:min(max_ext,length(exogenstg1))) {
                      for (j in 1:ncol(exogenstg1[[i]])) {
                        exogen[[h]] <- exogenstg1[[i]][,j]
                        h <- h+1
                      }
                    }
                } else {
                    exogen <- list(NULL)
                    exogen_future <- NULL
                }
                                
                data.ts <- cbind(data[[1]],cid)
                data.ts_train <- data.ts[1:(nrow(data)-test_periods),]
                data.ts_test <- data.ts[(nrow(data)-(test_periods-1)):nrow(data),]

                # create a grid of parameters for the best estimator for each series pair
                lags = seq(1,max_lags)
                include = c('none','const','trend','both')
                estim = c('2OLS','ML')

                grid <- expand.grid(lags = lags, include = include, estim = estim, exogen=exogen)
                grid$ic <- 999999

                for (i in 1:nrow(grid)){
                  if (is.null(grid[i,'exogen'][[1]])){
                    ex_train = NULL
                  } else {
                    ex_train = exogen_train[,grid[i,'exogen'][[1]]]
                  }

                  vc_train <- VECM(data.ts_train,
                                  r=r,
                                  lag=grid[i,'lags'],
                                  include = as.character(grid[i,'include']),
                                  estim = as.character(grid[i,'estim']),
                                  exogen=ex_train)
                  grid[i,'ic'] <-  IC(vc_train)
                }

                # choose parameters with best IC
                best_params <- grid[grid$ic == min(grid$ic),]

                # set externals
                if (is.null(best_params[1,'exogen'][[1]])){
                  ex_current = NULL
                  ex_future = NULL
                  ex_train = NULL
                  ex_test = NULL

                } else {
                  ex_current = data[,best_params[1,'exogen'][[1]]]
                  ex_future = exogen_future[,best_params[1,'exogen'][[1]]]
                  ex_train = exogen_train[,best_params[1,'exogen'][[1]]]
                  ex_test = exogen_test[,best_params[1,'exogen'][[1]]]
                }
                
                # predict on test set one more time with best parameters for model accuracy info
                vc_train <- VECM(data.ts_train,
                                  r=r,
                                  lag=best_params[1,'lags'],
                                  include = as.character(best_params[1,'include']),
                                  estim = as.character(best_params[1,'estim']),
                                  exogen=ex_train)
                p <- as.data.frame(predict(vc_train,n.ahead=test_periods,exoPred=ex_test))
                p$xreg <- as.character(best_params[1,'exogen'])[[1]]
                p$model_form <- paste0('VECM with ',
                                        best_params[1,'lags'],' lags',
                                        '|estimator: ',best_params[1,'estim'],
                                        '|include: ',best_params[1,'include'],
                                        '|selected regressors: ',as.character(best_params[1,'exogen'])[[1]])

                write.csv(p,'tmp/tmp_test_results.csv',row.names=F)

                # train the final model on full dataset with best parameter values
                vc.out = VECM(data.ts,
                              r=r,
                              lag=best_params[1,'lags'],
                              include = as.character(best_params[1,'include']),
                              estim = as.character(best_params[1,'estim']),
                              exogen=ex_current
                )

                # make the forecast
                f <- as.data.frame(predict(vc.out,n.ahead=n_ahead,exoPred=ex_future))
                # write final forecast values
                write.csv(f,'tmp/tmp_forecast.csv',row.names=F)

                summary_df <- t(coef(vc_train))
                write.csv(summary_df,'tmp/tmp_summary_output.csv')
        """)

        tmp_test_results = pd.read_csv('tmp/tmp_test_results.csv')
        tmp_forecast = pd.read_csv('tmp/tmp_forecast.csv')

        self.info[call_me]['holdout_periods'] = test_length
        self.info[call_me]['test_set_predictions'] = list(tmp_test_results.iloc[:,0])
        self.info[call_me]['test_set_actuals'] = self.y[(-test_length):]
        self.info[call_me]['test_set_ape'] = [np.abs(y - yhat) / np.abs(y) for y, yhat in zip(self.y[(-test_length):],tmp_test_results.iloc[:,0])]
        self.info[call_me]['model_form'] = tmp_test_results['model_form'][0]
        self.mape[call_me] = np.array(self.info[call_me]['test_set_ape']).mean()
        self.forecasts[call_me] = list(tmp_forecast.iloc[:,0])
        self.feature_importance[call_me] = pd.read_csv('tmp/tmp_summary_output.csv',index_col=0)

    def forecast_rf(self,test_length=1,Xvars='all',call_me='rf',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a random forest from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "rf"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict, default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argument collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = RandomForestRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Random Forest - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_gbt(self,test_length=1,Xvars='all',call_me='gbt',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a gradient boosting regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "rf"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict, default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argument collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.ensemble import GradientBoostingRegressor
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = GradientBoostingRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Gradient Boosted Trees - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_adaboost(self,test_length=1,Xvars='all',call_me='adaboost',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with an ada boost regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "rf"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict, default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argument collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.ensemble import AdaBoostRegressor
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = AdaBoostRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Ada Boosted Trees - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_mlp(self,test_length=1,Xvars='all',call_me='mlp',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a multi level perceptron neural network from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "mlp"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict, default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argument collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.neural_network import MLPRegressor
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = MLPRegressor(**hyper_params,random_state=20)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Multi Level Perceptron - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_mlr(self,test_length=1,Xvars='all',call_me='mlr',set_feature_importance=True):
        """ forecasts the stored y variable with a multi linear regression from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "mlr"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.linear_model import LinearRegression
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = LinearRegression()
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Multi Linear Regression')
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_ridge(self,test_length=1,Xvars='all',call_me='ridge',alpha=1.0,set_feature_importance=True):
        """ forecasts the stored y variable with a ridge regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "ridge"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        alpha : float, default 1.0
                            the desired alpha hyperparameter to pass to the sklearn model
                            1.0 is also the default in sklearn
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.linear_model import Ridge
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = Ridge(alpha=alpha)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Ridge - {}'.format(alpha))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_lasso(self,test_length=1,Xvars='all',call_me='lasso',alpha=1.0,set_feature_importance=True):
        """ forecasts the stored y variable with a lasso regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "lasso"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        alpha : float, default 1.0
                            the desired alpha hyperparameter to pass to the sklearn model
                            1.0 is also the default in sklearn
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.linear_model import Lasso
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = Lasso(alpha=alpha)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Lasso - {}'.format(alpha))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)

    def forecast_svr(self,test_length=1,Xvars='all',call_me='svr',hyper_params={},set_feature_importance=True):
        """ forecasts the stored y variable with a support vector regressor from sklearn (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
            Parameters: test_length : int, default 1
                            the length of the resulting test_set
                            must be at least 1 (AssertionError raised if not)
                        Xvars : list or "all", default "all"
                            the independent variables to use in the resulting X dataframes
                        call_me : str, default "mlp"
                            the model's nickname -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        hyper_params : dict, default {}
                            any hyper paramaters that you want changed from the default setting from sklearn, parameter is key, desired setting is value
                            passed as an argument collection (**hyper_params) to the sklearn model
                        set_feature_importance : bool or any other data type, default True
                            if True, adds a key to self.feature_importance with the call_me parameter as a key
                            value is the feature_importance dataframe from eli5 in a pandas dataframe data type
                            not setting this to True means it will be ignored, which improves speed
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        from sklearn.svm import SVR
        assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
        assert test_length >= 1, 'test_length must be at least 1'
        self.info[call_me] = dict.fromkeys(['holdout_periods','model_form','test_set_actuals','test_set_predictions','test_set_ape'])
        X, y, X_train, X_test, y_train, y_test = self._train_test_split(test_length=test_length,Xvars=Xvars)
        regr = SVR(**hyper_params)
        self._score_and_forecast(call_me,regr,X,y,X_train,y_train,X_test,y_test,Xvars)
        self._set_remaining_info(call_me,test_length,'Support Vector Regressor - {}'.format(hyper_params))
        if set_feature_importance:
            self.feature_importance[call_me] = self._set_feature_importance(X,y,regr)


    def forecast_average(self,models='all',exclude=None,call_me='average',test_length='max'):
        """ averages a set of models to make a new estimator
            Parameters: models : list, "all", or starts with "top_", default "all"
                            "all" will average all models
                            starts with "top_" will average the top however many models are specified according to their respective MAPE values on the test set (lower = better)
                                the character after "top_" must be an integer
                                ex. "top_5"
                            if list, then those are the models that will be averaged
                        exclude : list, default None
                            manually exlcude some models
                            all models passed here will be excluded
                            if models parameters starts with "top" and one of those top models is in the list passed to exclude, that model will be excluded 
                        call_me : str, default "average"
                            what to call the evaluated model -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        test_length : int or "max", default "max"
                            the test length to assign to the average model
                            if max, it will use the maximum test_length that all saved models can support
                            if int, will use that many test periods
                            if int is greater than one of the stored models' test length, this will fail
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        if models == 'all':
            avg_these_models = [e for e in list(self.mape.keys()) if (e != call_me) & (not e is None)]
        elif isinstance(models,list):
            avg_these_models = models[:]
        elif isinstance(models,str):
            if models.startswith('top_'):
                ordered_models = [e for e in self.order_all_forecasts_best_to_worst() if (e != call_me) & (not e is None)]
                avg_these_models = [m for i, m in enumerate(ordered_models) if (i+1) <= int(models.split('_')[1])]
        else:
            raise ValueError(f'argument in models parameter not recognized: {models}')

        if not exclude is None:
            if not isinstance(exclude,list):
                raise TypeError(f'exclude must be a list or None, not {type(exclude)}')
            else:
                avg_these_models = [m for m in avg_these_models if m not in exclude]
            
        if len(avg_these_models) == 0:
            print('no models found to average')
            return None

        if test_length == 'max':
            for i, m in enumerate(avg_these_models):
                if i == 0:
                    test_length = self.info[m]['holdout_periods']
                else:
                    test_length = min(test_length,self.info[m]['holdout_periods'])
        else:
            assert isinstance(test_length,int), f'test_length must be an int, not {type(test_length)}'
            assert test_length >= 1, 'test_length must be at least 1'

        self.mape[call_me] = 1
        self.forecasts[call_me] = [None]*self.forecast_out_periods

        self.info[call_me] = {'holdout_periods':test_length,
                             'model_form':None,
                             'test_set_actuals':self.y[-(test_length):],
                             'test_set_predictions':[None],
                             'test_set_ape':[None]}

        forecasts = pd.DataFrame()
        test_set_predictions_df = pd.DataFrame()
        test_set_ape_df = pd.DataFrame()
        for m in avg_these_models:
            test_set_predictions_df[m] = self.info[m]['test_set_predictions'][-(test_length):]
            test_set_ape_df[m] = self.info[m]['test_set_ape'][-(test_length):] 
            forecasts[m] = self.forecasts[m]
            
        self.info[call_me]['model_form'] = 'Average of ' + str(len(avg_these_models)) + ' models: ' + ', '.join(avg_these_models)
        self.info[call_me]['test_set_predictions'] = list(test_set_predictions_df.mean(axis=1))
        self.info[call_me]['test_set_ape'] = list(test_set_ape_df.mean(axis=1))
        self.mape[call_me] = np.array(self.info[call_me]['test_set_ape']).mean()
        self.forecasts[call_me] = list(forecasts.mean(axis=1))

    def forecast_naive(self,call_me='naive',mape=1.0):
        """ forecasts with a naive method of using the last observed y value propagated forward
            Parameters: call_me : str, default "naive"
                            what to call the evaluated model -- this name carries to the self.info, self.mape, and self.forecasts dictionaries
                        mape : float, default 1.0
                            the MAPE to assign to the model -- since the model is not tested, this should be some arbitrarily high number
                            if a numeric type is not passed, a value error is raised
            ***See forecast_auto_arima() documentation for an example of how to call a forecast method and access reults
        """
        self.mape[call_me] = float(mape)
        self.forecasts[call_me] = [self.y[-1]]*self.forecast_out_periods
        self.info[call_me] = {'holdout_periods':None,'model_form':'Naive','test_set_actuals':[None],'test_set_predictions':[None],'test_set_ape':[None]}

    def set_best_model(self):
        """ sets the best forecast model based on which model has the lowest MAPE value for the given holdout periods
            if two or more models tie, it will select whichever one was evaluated first
        """
        self.best_model = Counter(self.mape).most_common()[-1][0]

    def order_all_forecasts_best_to_worst(self):
        """ returns a list of the evaluated models for the given series in order of best-to-worst according to their evaluated MAPE values
            using different-sized test sets for different models could cause some trouble here, but I don't see a better way
        """
        x = [h[0] for h in Counter(self.mape).most_common()]
        return x[::-1] # reversed copy of the list

    def display_ts_plot(self,models='all',print_model_form=False,print_mapes=False):
        """ Plots time series results of the stored forecasts
            All models plotted in order of best-to-worst mapes
            Parameters: models : list, "all", or starts with "top_"; default "all"
                            the models you want plotted
                            if "all" plots all models
                            if list type, plots all models in the list
                            if starts with "top_" reads the next character(s) as the top however models you want plotted (based on lowest MAPE values)
                        print_model_form : bool, default False
                            whether to print the model form to the console of the models being plotted
                        print_mapes : bool, default False
                            whether to print the MAPEs to the console of the models being plotted
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        if isinstance(models,str):
            if models == 'all':
                plot_these_models = self.order_all_forecasts_best_to_worst()[:]
            elif models.startswith('top_'):
                top = int(models.split('_')[1])
                if top > len(self.forecasts.keys()):
                    plot_these_models = self.order_all_forecasts_best_to_worst()[:]
                else:
                    plot_these_models = self.order_all_forecasts_best_to_worst()[:top]
            else:
                raise ValueError(f'models argument not supported: {models}')
        elif isinstance(models,list):
            plot_these_models = [m for m in self.order_all_forecasts_best_to_worst() if m in models]
        else:
            raise ValueError(f'models must be list or str, got {type(models)}')

        if (print_model_form) | (print_mapes):
            for m in plot_these_models:
                print_text = '{} '.format(m)
                if print_model_form:
                    print_text += "model form: {} ".format(self.info[m]['model_form'])
                if print_mapes:
                    print_text += "{}-period test-set MAPE: {} ".format(self.info[m]['holdout_periods'],self.mape[m])
                print(print_text)

        sns.lineplot(x=pd.to_datetime(self.current_dates),y=self.y,ci=None)
        labels = ['Actual']

        for m in plot_these_models:
            sns.lineplot(x=pd.to_datetime(self.future_dates),y=self.forecasts[m])
            labels.append(m)

        plt.legend(labels=labels,loc='best')
        plt.xlabel('Date')
        plt.ylabel(f'{self.name}')
        plt.title(f'{self.name} Forecast Results')
        plt.show()

