import os.path, time
from tqdm import tqdm
import pandas as pd
import os
import omegaconf
from typing import List, Literal, Union, Optional
import yaml


def unroll_to_dict(cfg:omegaconf.OmegaConf, level=[]) -> dict:
  """unroll hierarchial dict to flat dict. All returned values are strings"""
  flat_cfg={}
  for k,v in cfg.items():
    # instance can be DictConfig (if read via hydra) or dict (if read via yaml)
    if isinstance(v,omegaconf.dictconfig.DictConfig) or isinstance(v,dict):
      flat_cfg.update(unroll_to_dict(v,level + [k]))
    else:
      flat_k = ".".join(level + [k])
      flat_cfg[flat_k] = str(v) # make it into a string; Lightning App requires string
  return(flat_cfg)

def get_configs_in_dir(hydra_search_dir: str, config_name: str = "config.yaml") -> List[str]:
    """
    Find all the saved config files in the directory
    """
    assert os.path.isdir(hydra_search_dir)
    filelist = []
    for dirpath, dirnames, filenames in os.walk(hydra_search_dir):
        for filename in [f for f in filenames if f.endswith(config_name)]:
            filelist.append(os.path.join(dirpath, filename))
    return filelist

def remove_quotes(s: str) -> str:
    return s.replace("'", "")

def model_path(config_file: str) -> str:
    return "/".join(config_file.split("/")[:-2])

class ModelInventoryBuilder:
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.config_list = get_configs_in_dir(self.config_dir)
        # now eliminate configs w/o associated predictions.csv
        self.config_list = [f for f in self.config_list if os.path.isfile(os.path.join(model_path(f), "predictions.csv"))]
        self.df = pd.DataFrame()
    
    @property
    def num_configs(self) -> int:
        return len(self.config_list)
    
    @property
    def model_paths(self) -> List[str]:
        return [model_path(f) for f in self.config_list]
    
    def build_dframe(self) -> pd.DataFrame:
        print("Building model registry from %i configs..." % self.num_configs)
        for i, config_file in enumerate(tqdm(self.config_list)):
            # read yaml
            with open(config_file, "rb") as f:
                cfg_ = yaml.safe_load(f)
            flat_dict = unroll_to_dict(cfg_) # flatten hierarchy
            df_ = pd.DataFrame([flat_dict]) # make as df
            time_created = time.ctime(os.path.getctime(config_file)) # get time created
            # print time created before and after timestamp
            # print(time_created)
            df_["timestamp"] = pd.Timestamp(time_created) # add timestamp
            df_["path"] = self.model_paths[i] # add path to artifacts
            self.df = pd.concat([self.df, df_]) # add to total df
        # remove quotes from losses to use, it may interfere with queries
        if "model.losses_to_use" in self.df.columns:
            self.df["model.losses_to_use"] = self.df["model.losses_to_use"].apply(remove_quotes)
        
        return self.df
     
class QueryBuilder:
    """ Useful string operations to build pandas queries that obey hydra + pandas + lightning syntax """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.query_list = []
        # self.value_list = [] # we need this for @ operators
    
    # each query has a parameter name, condition, and value or list of values
    def add_query(self, parameter_name: str, \
        condition: Literal[">", "<", "==", "!=", "in", "not in"], 
        value: str) -> None:

        # self.value_list.append(value)
        
        # make sure that people don't err with adding double quotes
        if parameter_name == "model.losses_to_use" and isinstance(value, str):
            value = remove_quotes(value)
        
        if condition == "in" or condition == "not in":
            # assert that value is a list of strs
            assert isinstance(value, list), "value must be a list"
            if condition == "in":
                # this line manually implements the @ operator
                self.query_list.append(" or ".join([f"`{parameter_name}` == '{v}'" for v in value]))
                print(self.query_list[-1])          
        # note the use of single quotes for the value and backticks for the parameter name which includes periods
        else: 
            self.query_list.append(f"`{parameter_name}` {condition} '{value}'")
    
    def add_timestamp_query(self, start_date: str, end_date: str) -> None:
        if start_date is not None and end_date is not None:
            self.query_list.append(f"timestamp >= '{start_date}' and timestamp <= '{end_date}'")
        elif start_date is not None:
            self.query_list.append(f"timestamp >= '{start_date}'")
        elif end_date is not None:
            self.query_list.append(f"timestamp <= '{end_date}'")
        elif start_date is None and end_date is None:
            pass

    
    def combine_queries(self, operator: Literal["and", "or"]) -> str:
        # add operator between each query str
        # currently supports a single combination operator
        return (" %s " % operator).join([f"({q})" for q in self.query_list])
        
    def get_query(self) -> pd.DataFrame:
        return self.df.query(self.query_dict)


def load_metric_csv(
        filename: str, metric_name: str, split_set: Optional[str] = None, pd_kwargs: dict = {},
) -> pd.DataFrame:
    df = pd.read_csv(filename, **pd_kwargs)
    # take mean over all columns that are not "set"
    df["mean"] = df.loc[:, [c for c in df.columns if c != "set"]].mean(axis=1)
    if split_set is not None:
        df["set"] = split_set
    # add mean and metric
    df["metric"] = metric_name
    return df


def load_confidence_csv(filename):
    df = pd.read_csv(filename, header=[1, 2], index_col=0)
    vals = df.to_numpy()
    likelihoods = vals[:, 2::3]
    splits = vals[:, -1]
    keypoint_names = df.columns.get_level_values(0)[2::3]
    df_confs = pd.DataFrame(likelihoods, columns=keypoint_names, index=df.index)
    df_confs["mean"] = df_confs.mean(axis=1)
    if vals.shape[1] % 3 == 1:
        # labeled data
        df_confs["set"] = splits
    df_confs["metric"] = "confidence"
    return df_confs


def get_model_type(row: Union[pd.Series, dict]) -> str:
    if row["model.model_type"] == "heatmap":
        if row["model.losses_to_use"] == "[]":
            model_type = "baseline"
        else:
            model_type = "semi-super"
    elif row["model.model_type"] == "heatmap_mhcrnn":
        if row["model.losses_to_use"] == "[]":
            model_type = "context"
        else:
            model_type = "semi-super context"
    elif row["model.model_type"] == "dlc":
        model_type = "dlc"
    return model_type
