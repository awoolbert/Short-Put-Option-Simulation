import os
import time
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from datetime import datetime, timedelta
from typing import List, Tuple
from scipy.stats import norm
from dataclasses import dataclass

OPTIONS_DATA_FNAME = 'raw_options_data.csv'

def get_thesis_path() -> str:
    """Locates the Thesis folder in Dropbox

    Returns:
        str: full path description for Thesis folder
    """
    start_path = os.path.dirname(os.path.realpath(__file__))
    found_it = False
    temp_path = start_path
    while not found_it:
        if temp_path[-7:] == 'Dropbox':
            found_it = True
        else:
            temp_path = os.path.dirname(temp_path)
    avery_files = os.path.join(temp_path, 'Avery Files')
    return os.path.join(avery_files, 'Thesis')
THESIS_PATH = get_thesis_path()

def get_spx_data() -> Series:
    """Retrieves the SPX data from the csv file

    Returns:
        Series: SPX data
    """
    spx = pd.read_csv(os.path.join(THESIS_PATH, 'Data', 'spx.csv'), index_col=0)
    spx.index = pd.to_datetime(spx.index)
    return spx
spx = get_spx_data()

def get_vix_data() -> Series:
    """Retrieves the VIX data from the csv file

    Returns:
        Series: VIX data
    """
    vix = pd.read_csv(os.path.join(THESIS_PATH, 'Data', 'vix.csv'), index_col=0)
    vix.index = pd.to_datetime(vix.index)
    return vix
vix = get_vix_data()

def get_sptr_data() -> Series:
    """Retrieves the SPTR data from the csv file

    Returns:
        Series: SPTR data
    """
    fpath = os.path.join(THESIS_PATH, 'Data', 'sptr.csv')
    sptr = pd.read_csv(fpath, index_col=0)
    sptr.index = pd.to_datetime(sptr.index)
    return sptr
sptr = get_sptr_data()

def get_tbills_data() -> Series:
    """Retrieves the T-bills data from the csv file

    Returns:
        Series: T-bills data
    """
    fpath = os.path.join(THESIS_PATH, 'Data', 'tbills.csv')
    tbills = pd.read_csv(fpath, index_col=0)
    tbills.index = pd.to_datetime(tbills.index)
    return tbills
tbills = get_tbills_data()

def get_spx_option_settlement_dates() -> List[datetime]:
    """Retrieves the settlement dates SPX options

    Returns:
        List[datetime]: list of settlement dates
    """
    # get list of ever third Friday of the month since 1/1/1996
    settle_dts = pd.date_range(
        start='1/1/1996', end='9/30/2023', freq='WOM-3FRI'
    ).to_list()

    # find third fridays where spx did not trade
    holidays = [date for date in settle_dts if date not in spx.index]

    # replace each holiday with day before
    for holiday in holidays:
        settle_dts[settle_dts.index(holiday)] = holiday - timedelta(days=1)

    return settle_dts
settle_dts = get_spx_option_settlement_dates()

class OptionsDataAssembler:

    def __init__(self, use_muted_vix: bool = False) -> None:
        self.use_muted_vix = use_muted_vix
        self.res_fpath = os.path.join(THESIS_PATH, 'Data', 'options_data.csv')
        self.spx = spx
        self.vix = vix
        self.tbills = tbills
        self.sptr = sptr
        self.settle_dts = settle_dts

    def run(self, zs_out_of_money: float = 1) -> DataFrame:
        """Assmeble DataFrame of options used in strategy

        Args:
            zs_out_of_money (float, optional): how many z-scores out of the 
                money to sell the option (using VIX to determine vol).
                Defaults to 1.
        """
        print('Running OptionsDataAssembler')
        tgt_options = self.get_desired_options_list(zs_out_of_money)
        data = self.load_data_from_database()
        data = self.limit_data_to_desired_options(data, tgt_options)
        data = self.append_cols_needed_for_bs(data)
        data = self.calculate_premiums(data)
        self.save_data(data)

    def get_desired_options_list(self, zs: float) -> List[Tuple[datetime, int]]:
        """Retrieves the list of SPX options in the strategy

        Returns:
            List[Tuple[datetime, int]]: list of SPX options in the strategy where
                each tuple is (settlement date, strike price)
        """
        print('  creating list of options for strategy')
        options = []
        for idx, dt in enumerate(settle_dts[:-1]):
            # identify dt on which option selection is being made
            selection_dt = dt - timedelta(days=2)
            if selection_dt not in self.spx.index:
                selection_dt = dt - timedelta(days=1)

            # get spx closing price on that dt
            spx_level = self.spx.loc[selection_dt].values[0]

            # get vix closing level on that dt
            if self.use_muted_vix:
                vix_level = self.vix.loc[selection_dt].values[0] ** 0.5
                vix_level = (vix_level * 4) / 100
            else:
                vix_level = self.vix.loc[selection_dt].values[0] / 100

            # deterimine target level (1z below current price)
            monthly_sd = vix_level / 12**(0.5)
            tgt_level = round(spx_level * (1 - monthly_sd * zs), -1)

            # get corresponding expiration date
            exp_dt = self.settle_dts[idx + 1]

            options.append((exp_dt, int(tgt_level)))
        return options

    def load_data_from_database(self) -> DataFrame:
        """Retrieves the SPX option data from the csv file

        Returns:
            DataFrame: SPX option data
        """
        print('  loading database into memory')
        fpath = os.path.join(THESIS_PATH, 'Data', OPTIONS_DATA_FNAME)
        dtype_dict = {
            'expiry_indicator': 'str',
            'root': 'str',
            'suffix': 'str'
        }
        df = pd.read_csv(fpath, dtype=dtype_dict)

        # convert columns to appropriate data types
        for col in ['date', 'exdate']:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')

        # remove weekly options
        df = df[df['symbol'].str.contains('SPXW') == False]

        # adjust settle dates and limit to only those in strategy
        tgt_dts_plus_1 = [d + timedelta(days=1) for d in self.settle_dts]
        tgt_dts_plus_2 = [d + timedelta(days=2) for d in self.settle_dts]
        all_dts = self.settle_dts + tgt_dts_plus_1 + tgt_dts_plus_2
        df = df[df['exdate'].isin(all_dts)]
        df['exdate'] = [
            d if d in self.settle_dts 
            else d - timedelta(days=1) if d in tgt_dts_plus_1 
            else d - timedelta(days=2)
            for d in df['exdate']
        ]
        missing = [d for d in self.settle_dts if d not in df['exdate'].values]
        if missing:
            print(f'Missing {len(missing)} settlement dates: {missing}')

        # convert strike price to integer
        df['strike_price'] = (df['strike_price'] / 1000).astype(int)
        df['id'] = df['exdate'].astype(str) + '_' + df['strike_price'].astype(str)
        return df

    def limit_data_to_desired_options(self, 
        df: DataFrame, 
        tgt_options: List[Tuple[datetime, int]]
    ) -> DataFrame:
        """Limits the SPX options data to only those in the strategy

        Args:
            df (DataFrame): SPX options data

        Returns:
            DataFrame: SPX options data limited to only those in the strategy
        """
        print('  extracting options data from database')
        tgt_option_strs = [
            f'{dt:%Y-%m-%d}_{strike}' for dt, strike in tgt_options
        ]

        used_options = []
        for opt_str in tgt_option_strs:
            orig_opt_str = opt_str
    
            # ensure sufficient data for option
            is_found = False
            max_attempts = 30
            while not is_found:
                opt_df = df[df['id'] == opt_str]
                if len(opt_df) > 22:
                    is_found = True
                    used_options.append(opt_str)
                    if opt_str == orig_opt_str:
                        print(f'  Found {opt_str}')
                    else:
                        print(f'  {orig_opt_str} -> {opt_str}')
                else:
                    if max_attempts == 0:
                        print(f'  Could not find {opt_str} or any alternatives')
                        break
                    dt, strike = opt_str.split('_')
                    new_strike = int(int(strike) + 10)
                    opt_str = f'{dt}_{new_strike}'
                    max_attempts -= 1
        df = df[df['id'].isin(used_options)]

        # limit columns
        cols = [
        'id', 'date', 'symbol', 'exdate', 'strike_price', 'best_bid', 'best_offer', 
        'volume', 'open_interest', 'impl_volatility', 'delta', 'optionid'
        ]
        df = df[cols]

        # calculate days to expiry
        df['days'] = (df['exdate'] - df['date']).dt.days
        df = df[df['days'] >= 0 & (df['days'] < 40)]
        
        return df

    def old_limit_data_to_desired_options(self, 
        df: DataFrame, 
        tgt_options: List[Tuple[datetime, int]]
    ) -> DataFrame:
        """Limits the SPX options data to only those in the strategy

        Args:
            df (DataFrame): SPX options data

        Returns:
            DataFrame: SPX options data limited to only those in the strategy
        """
        print('  extracting options data from database')
        tgt_option_strs = [
            f'{dt:%Y-%m-%d}_{strike}' for dt, strike in tgt_options
        ]

        # for any missing options, attempt to find alternatives
        missing = [opt for opt in tgt_option_strs if opt not in df['id'].values]
        found = []
        for opt in missing:
            for i in range(1, 10):
                dt, strike = opt.split('_')
                new_strike = int(int(strike) + i*10)
                if f'{dt}_{new_strike}' in df['id'].values:
                    tgt_option_strs.append(f'{dt}_{new_strike}')
                    found.append(opt)
                    break
        missing = [opt for opt in missing if opt not in found]
        if missing:
            print(f'Missing {len(missing)} options: {missing}')
        df = df[df['id'].isin(tgt_option_strs)]

        # limit columns
        cols = [
        'id', 'date', 'symbol', 'exdate', 'strike_price', 'best_bid', 'best_offer', 
        'volume', 'open_interest', 'impl_volatility', 'delta', 'optionid'
        ]
        df = df[cols]

        # calculate days to expiry
        df['days'] = (df['exdate'] - df['date']).dt.days
        df = df[df['days'] >= 0 & (df['days'] < 40)]
        
        return df

    def append_cols_needed_for_bs(self, df: DataFrame) -> DataFrame:
        """Appends columns needed for the Black-Scholes model

        Args:
            df (DataFrame): SPX options data

        Returns:
            DataFrame: SPX options data with additional columns needed for the
                Black-Scholes model
        """
        print('  appending columns for black-scholes calculations')
        # calculate time to expiry in years
        df['t'] = df['days'] / 365

        # calculate risk-free rate by using tbills data
        tbills_col = pd.merge(
            df[['date']], self.tbills, left_on='date', right_index=True, how='left'
        )
        df['r'] = tbills_col['usgg3m index'] / 100

        # calculate stock price by using spx data
        spx_col = pd.merge(
            df[['date']], self.spx, left_on='date', right_index=True, how='left'
        )

        # select remaining needed columns
        df['s'] = spx_col['spx index']
        df['k'] = df['strike_price']
        df['sigma'] = df['impl_volatility']

        return df   

    def calculate_premiums(self, df: DataFrame) -> DataFrame:
        """Calculates the premiums of the options

        Args:
            df (DataFrame): SPX options data

        Returns:
            DataFrame: SPX options data with premiums calculated
        """
        print('  calculating premiums using black-scholes')
        # calculate the premiums
        df['premium'] = df.apply(
            lambda x: self.black_scholes(
                x['s'], x['k'], x['t'], x['r'], x['sigma'], -1
            ),
            axis=1
        )
        return df

    def black_scholes(self,
        s: float, 
        k: float, 
        t: float, 
        r: float, 
        sigma: float, 
        call_put: int
    ) -> float:
        """Calculates the Black-Scholes option price

        Args:
            s (float): stock price
            k (float): strike price
            t (float): time to expiry in years
            r (float): risk-free rate
            sigma (float): volatility
            call_put (int): 1 for call, -1 for put

        Returns:
            float: Black-Scholes option price
        """
        d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * t**0.5)
        d2 = d1 - sigma * t**0.5
        return (
            call_put * s * norm.cdf(call_put * d1) 
            - call_put * k * np.exp(-r * t) * norm.cdf(call_put * d2)
        )

    def save_data(self, df: DataFrame) -> None:
        print('  saving data')
        fpath = os.path.join(THESIS_PATH, 'Data', self.res_fpath)
        df.to_csv(fpath, index=False)
        # expiry_indicator	root	suffix

@dataclass
class SimRow:
    date: datetime
    equity_beg: float
    pos: float
    premium: float
    pnl_pos: float
    is_close: int
    close_out_price: float
    pnl_cover: float
    is_open: int
    open_price: float
    cash: float
    interest: float
    pnl: float
    equity_end: float
    option: str
    spx: float
    vix: float

class ShortOptionSimulationRunner:

    def __init__(self, leverage: float = 1) -> None:
        self.leverage = leverage
        self.data_fpath = os.path.join(THESIS_PATH, 'Data', 'options_data.csv')
        self.spx = spx
        self.tbills = tbills
        self.sptr = sptr
        self.vix = vix
        self.settle_dts = settle_dts
        self.df: Series = None
        self.data: DataFrame = None
        self.options: List[str] = None
        self.simulation: DataFrame = None
        self.rets: DataFrame = None

    def run(self) -> None:
        print('Running ShortOptionSimulationRunner')
        self._load_data()
        self._forward_fill_series_to_ensure_no_missing_dates()
        self._set_options_list()
        self._step_through_dates_to_simulate_pnl()
        self._save_simulation()
        self._calc_monthly_returns()
        self._save_monthly_returns()

    def _load_data(self) -> None:
        """Loads the SPX options data from the csv file

        Returns:
            DataFrame: SPX options data
        """
        print('  loading data')
        df = pd.read_csv(self.data_fpath)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['exdate'] = pd.to_datetime(df['exdate'], format='%Y-%m-%d')
        self.data = df

    def _forward_fill_series_to_ensure_no_missing_dates(self) -> None:
        """Forward fills the series to ensure no missing dates"""
        dates = pd.date_range(
            self.data['date'].min(), self.data['date'].max(), freq='B'
        )
        self.spx = self.spx.reindex(dates, method='ffill')
        self.tbills = self.tbills.reindex(dates, method='ffill')
        self.vix = self.vix.reindex(dates, method='ffill')
        self.sptr = self.sptr.reindex(dates, method='ffill')

    def _set_options_list(self) -> None:
        """Retrieves the list of SPX options in the strategy

        Args:
            data (DataFrame): SPX options data

        Returns:
            List[str]: list of SPX options in the strategy
        """
        print('  creating list of options for strategy')
        options = self.data['id'].unique()
        options.sort()
        self.options = options

    def _step_through_dates_to_simulate_pnl(self) -> None:
        """Steps through the dates to simulate the P&L of the strategy"""
        options_idx, settle_dts_idx = 0, 0
        sim_rows: List[SimRow] = []
        equity = 100
        sim_dts = pd.to_datetime(self.data['date'].unique())

        for idx, dt in enumerate(sim_dts):
            # record beginning equity
            equity_beg = equity

            # if first day, set initial values
            if idx == 0:
                pos, pnl_pos, premium, interest_days = 0, 0, 0, 1
                cash = equity
                is_close, is_open = 0, 1  # trade on first day to open position
                option = None

            else:
                prev = sim_rows[-1] # previous day's values

                # determine how many days interest should be based on
                interest_days = (dt - sim_dts[idx - 1]).days

                # update starting values based on previous day ending values
                option = prev.option
                pos = prev.pos

                # get the closing price for the current option
                premium = self._get_price(option, dt)
                
                # determine prev price (using open price if traded yesterday)
                just_traded = prev.is_open == 1
                prev_premium = prev.open_price if just_traded else prev.premium
                
                # calculate pnl from position
                pnl_pos = pos * (premium - prev_premium)

                # determine if trade should be made
                next_settle_dt = self.settle_dts[settle_dts_idx]
                is_close = 1 if dt == next_settle_dt - timedelta(days=1)  else 0
                is_open = 1 if dt >= next_settle_dt else 0

            if is_close:
                trade = -pos
                pos = 0
                close_out_price = self._get_price(option, dt, 'best_offer')

                # calculate pnl from closing out position. comparing to
                # premium because previous -> premium was used for pnl_pos
                pnl_cover = trade * (premium - close_out_price)

            elif is_open:
                # open new position
                option = self.options[options_idx]
                open_price = self._get_price(option, dt, 'best_bid')
                spx_price = self.spx.loc[dt].values[0]
                pos = -1 * equity / spx_price * self.leverage
                close_out_price, pnl_cover = 0, 0

                # increment indices to next option and settle date
                settle_dts_idx += 1
                options_idx += 1

            else:
                pnl_cover = 0
                close_out_price = 0
                open_price = 0

            # calculate pnl from interest
            cash_rate = self.tbills.loc[dt].values[0] / 100
            interest = cash * cash_rate * interest_days / 365
            if interest_days > 4:
                print(f' Found missing data for {dt}...{interest_days} days')

            # calculate total pnl and update equity
            pnl = pnl_pos + pnl_cover + interest
            equity += pnl
            cash = equity

            # record values for this date
            vix_level = self.vix.loc[dt].values[0]
            spx_level = self.spx.loc[dt].values[0]
            simr = SimRow(
                dt, equity_beg, pos, premium, pnl_pos, is_close, 
                close_out_price, pnl_cover, is_open, open_price, cash, interest, 
                pnl, equity, option, spx_level, vix_level
            )
            sim_rows.append(simr)
        
        self.simulation = self._build_df_from_sim_rows(sim_rows)
        print(self.simulation)

    def _get_price(self, 
        option: str, 
        dt: datetime, 
        field: str = 'best_bid'
    ) -> float:
        """Retrieves the premium for the option on the date

        Args:
            option (str): option
            dt (datetime): date

        Returns:
            float: premium
        """
        is_option = self.data['id'] == option
        is_dt = self.data['date'] == dt
        df = self.data.loc[is_option & is_dt, field]
        if df.empty:
            dt_str = dt.strftime('%Y-%m-%d')
            option_dt_str = option.split('_')[0]
            if dt_str != option_dt_str:
                print(f'  No premium found for {option} on {dt_str}. Using X-S')
            s = self.spx.loc[dt].values[0]
            x = int(option.split('_')[1])
            return max(x - s, 0)
        return df.values[0]

    def _build_df_from_sim_rows(self, sim_rows: List[SimRow]) -> DataFrame:
        """Builds a DataFrame from the simulation rows

        Args:
            sim_rows (List[SimRow]): simulation rows

        Returns:
            DataFrame: simulation data
        """
        df = DataFrame([vars(sr) for sr in sim_rows])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df

    def _save_simulation(self) -> None:
        """Saves the simulation data to a csv file"""
        print('  saving simulation')
        fpath = os.path.join(THESIS_PATH, 'Data', 'simulation.csv')
        self.simulation.to_csv(fpath)

    def _calc_monthly_returns(self) -> None:
        """Calculates the monthly returns"""
        print('  calculating monthly returns')
        sim_monthly_vals = self.simulation['equity_end'].resample('M').last()
        sim_rets = sim_monthly_vals.pct_change()

        idx = sim_monthly_vals.index
        sptr_monthly_vals = self.sptr['sptr index'].resample('M').last().loc[idx]
        sptr_rets = sptr_monthly_vals.pct_change()

        df = self.tbills / 100
        df['days_between'] = df.index.to_series().diff().dt.days
        df['cash_ret'] = df['usgg3m index'] / 365 * df['days_between']
        df = df['cash_ret']
        cash_rets = df.resample('M').sum().loc[idx]

        self.rets = pd.concat([sim_rets, sptr_rets, cash_rets], axis=1)
        self.rets.columns = ['strategy', 'sptr', 'cash']

    def _save_monthly_returns(self) -> None:
        """Saves the monthly returns to a csv file"""
        print('  saving monthly returns')
        fpath = os.path.join(THESIS_PATH, 'Data', 'monthly_returns.csv')
        self.rets.to_csv(fpath)

if __name__ == '__main__':
    t0 = time.time()
    # OptionsDataAssembler(use_muted_vix=True).run()
    ShortOptionSimulationRunner().run()
    print(f'Elapsed time: {time.time() - t0:.2f} seconds')