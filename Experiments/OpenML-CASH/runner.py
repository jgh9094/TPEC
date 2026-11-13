# run the CASH evolutionary algorithm (evolves full scikit-learn pipelines) on every dataset in an OpenML suite folder

import sys
import os
import csv
import traceback
import pandas as pd
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
sys.path.insert(0, REPO_ROOT)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../Data-Preprocessing')))

import Source.CASH.ea as optimizer
import argparse
import ray
from pathlib import Path
from typing import List, Tuple

# components (get_model_type() names) the evolution may use at each pipeline node: every
# component in Source/ML except PolynomialFeatures and Nystroem. PolynomialFeatures and Nystroem are
# excluded for run time: Nystroem's n_components can reach the number of training samples (thousands of
# features), which made GradientBoosting pipelines stall. "passthrough" is always available where the
# node allows it, so it is not listed.
SCALERS = ['MinMaxScaler', 'RobustScaler', 'StandardScaler', 'MaxAbsScaler', 'Normalizer']
TRANSFORMERS = ['KBinsDiscretizer', 'Binarizer', 'PCA', 'FastICA', 'FeatureAgglomeration',
                'RBFSampler', 'QuantileTransformer', 'PowerTransformer']
SELECTORS = ['SelectFwe', 'SelectPercentile', 'VarianceThreshold']
CLASSIFIERS = ['RF', 'ET', 'KSVC', 'GB', 'KNN', 'MLP', 'XGB']


def load_suite_targets(suite_directory: Path) -> List[Tuple[int, str]]:
    """Return (task_id, target_name) for every dataset in the suite's tasks_summary.csv, in file order."""
    summary_path = suite_directory / 'tasks_summary.csv'
    if not summary_path.is_file():
        raise FileNotFoundError(f"Suite summary does not exist: {summary_path}")
    with open(summary_path, newline='') as f:
        return [(int(row['task_id']), row['target_name']) for row in csv.DictReader(f)]


# Feature columns for each suite task, as load_task_data passes them to the CASH EA. one_hot_cols are
# one-hot encoded; scalar_cols pass through as numeric. Together they list every feature column of the
# task's CSV exactly once. Taken from the categorical indicators, matched to attribute names by
# write_attribute_names.py. load_task_data checks these lists against the CSV before use.
TASK_FEATURE_COLUMNS = {
    146818: {
        'one_hot_cols': [
            'A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12',
        ],
        'scalar_cols': [
            'A2', 'A3', 'A7', 'A10', 'A13', 'A14',
        ],
    },
    359955: {
        'one_hot_cols': [],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4',
        ],
    },
    190146: {
        'one_hot_cols': [],
        'scalar_cols': [
            'COMPACTNESS', 'CIRCULARITY', 'DISTANCE_CIRCULARITY', 'RADIUS_RATIO',
            'PR.AXIS_ASPECT_RATIO', 'MAX.LENGTH_ASPECT_RATIO', 'SCATTER_RATIO', 'ELONGATEDNESS',
            'PR.AXIS_RECTANGULARITY', 'MAX.LENGTH_RECTANGULARITY', 'SCALED_VARIANCE_MAJOR',
            'SCALED_VARIANCE_MINOR', 'SCALED_RADIUS_OF_GYRATION', 'SKEWNESS_ABOUT_MAJOR',
            'SKEWNESS_ABOUT_MINOR', 'KURTOSIS_ABOUT_MAJOR', 'KURTOSIS_ABOUT_MINOR', 'HOLLOWS_RATIO',
        ],
    },
    168757: {
        'one_hot_cols': [
            'checking_status', 'credit_history', 'purpose', 'savings_status', 'employment',
            'personal_status', 'other_parties', 'property_magnitude', 'other_payment_plans',
            'housing', 'job', 'own_telephone', 'foreign_worker',
        ],
        'scalar_cols': [
            'duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age',
            'existing_credits', 'num_dependents',
        ],
    },
    359956: {
        'one_hot_cols': [],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
            'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
            'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
            'V39', 'V40', 'V41',
        ],
    },
    359958: {
        'one_hot_cols': [],
        'scalar_cols': [
            'LOC_BLANK', 'BRANCH_COUNT', 'CALL_PAIRS', 'LOC_CODE_AND_COMMENT', 'LOC_COMMENTS',
            'CONDITION_COUNT', 'CYCLOMATIC_COMPLEXITY', 'CYCLOMATIC_DENSITY', 'DECISION_COUNT',
            'DECISION_DENSITY', 'DESIGN_COMPLEXITY', 'DESIGN_DENSITY', 'EDGE_COUNT',
            'ESSENTIAL_COMPLEXITY', 'ESSENTIAL_DENSITY', 'LOC_EXECUTABLE', 'PARAMETER_COUNT',
            'HALSTEAD_CONTENT', 'HALSTEAD_DIFFICULTY', 'HALSTEAD_EFFORT', 'HALSTEAD_ERROR_EST',
            'HALSTEAD_LENGTH', 'HALSTEAD_LEVEL', 'HALSTEAD_PROG_TIME', 'HALSTEAD_VOLUME',
            'MAINTENANCE_SEVERITY', 'MODIFIED_CONDITION_COUNT', 'MULTIPLE_CONDITION_COUNT',
            'NODE_COUNT', 'NORMALIZED_CYLOMATIC_COMPLEXITY', 'NUM_OPERANDS', 'NUM_OPERATORS',
            'NUM_UNIQUE_OPERANDS', 'NUM_UNIQUE_OPERATORS', 'NUMBER_OF_LINES', 'PERCENT_COMMENTS',
            'LOC_TOTAL',
        ],
    },
    359959: {
        'one_hot_cols': [
            'Wifes_education', 'Husbands_education', 'Wifes_religion', 'Wifes_now_working%3F',
            'Husbands_occupation', 'Standard-of-living_index', 'Media_exposure',
        ],
        'scalar_cols': [
            'Wifes_age', 'Number_of_children_ever_born',
        ],
    },
    2073: {
        'one_hot_cols': [],
        'scalar_cols': [
            'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc',
        ],
    },
    359960: {
        'one_hot_cols': [
            'buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety',
        ],
        'scalar_cols': [],
    },
    168784: {
        'one_hot_cols': [],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
            'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
            'V27',
        ],
    },
    359962: {
        'one_hot_cols': [],
        'scalar_cols': [
            'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't', 'lOCode',
            'lOComment', 'lOBlank', 'locCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op',
            'total_Opnd', 'branchCount',
        ],
    },
    359963: {
        'one_hot_cols': [],
        'scalar_cols': [
            'short.line.density.5', 'short.line.density.2', 'vedge.mean', 'vegde.sd', 'hedge.mean',
            'hedge.sd', 'intensity.mean', 'rawred.mean', 'rawblue.mean', 'rawgreen.mean',
            'exred.mean', 'exblue.mean', 'exgreen.mean', 'value.mean', 'saturation.mean',
            'hue.mean',
        ],
    },
    190137: {
        'one_hot_cols': [],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
            'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
            'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
            'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50',
            'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62',
            'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72',
        ],
    },
    168911: {
        'one_hot_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V14', 'V15',
            'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V24', 'V25', 'V26', 'V27', 'V28',
            'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40',
            'V41', 'V42', 'V44', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54',
            'V55', 'V57', 'V58', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68',
            'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80',
            'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92',
            'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104',
            'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115',
            'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V127',
            'V128', 'V129', 'V130', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139',
            'V140', 'V141', 'V142', 'V143', 'V144',
        ],
        'scalar_cols': [
            'V13', 'V23', 'V43', 'V45', 'V56', 'V59', 'V126', 'V131',
        ],
    },
    359964: {
        'one_hot_cols': [
            'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13',
            'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25',
            'A26', 'A27', 'A28', 'A29', 'A30', 'A31', 'A32', 'A33', 'A34', 'A35', 'A36', 'A37',
            'A38', 'A39', 'A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A47', 'A48', 'A49',
            'A50', 'A51', 'A52', 'A53', 'A54', 'A55', 'A56', 'A57', 'A58', 'A59', 'A60', 'A61',
            'A62', 'A63', 'A64', 'A65', 'A66', 'A67', 'A68', 'A69', 'A70', 'A71', 'A72', 'A73',
            'A74', 'A75', 'A76', 'A77', 'A78', 'A79', 'A80', 'A81', 'A82', 'A83', 'A84', 'A85',
            'A86', 'A87', 'A88', 'A89', 'A90', 'A91', 'A92', 'A93', 'A94', 'A95', 'A96', 'A97',
            'A98', 'A99', 'A100', 'A101', 'A102', 'A103', 'A104', 'A105', 'A106', 'A107', 'A108',
            'A109', 'A110', 'A111', 'A112', 'A113', 'A114', 'A115', 'A116', 'A117', 'A118', 'A119',
            'A120', 'A121', 'A122', 'A123', 'A124', 'A125', 'A126', 'A127', 'A128', 'A129', 'A130',
            'A131', 'A132', 'A133', 'A134', 'A135', 'A136', 'A137', 'A138', 'A139', 'A140', 'A141',
            'A142', 'A143', 'A144', 'A145', 'A146', 'A147', 'A148', 'A149', 'A150', 'A151', 'A152',
            'A153', 'A154', 'A155', 'A156', 'A157', 'A158', 'A159', 'A160', 'A161', 'A162', 'A163',
            'A164', 'A165', 'A166', 'A167', 'A168', 'A169', 'A170', 'A171', 'A172', 'A173', 'A174',
            'A175', 'A176', 'A177', 'A178', 'A179',
        ],
        'scalar_cols': [],
    },
    359965: {
        'one_hot_cols': [
            'bkblk', 'bknwy', 'bkon8', 'bkona', 'bkspr', 'bkxbq', 'bkxcr', 'bkxwp', 'blxwp',
            'bxqsq', 'cntxt', 'dsopp', 'dwipd', 'hdchk', 'katri', 'mulch', 'qxmsq', 'r2ar8',
            'reskd', 'reskr', 'rimmx', 'rkxwp', 'rxmsq', 'simpl', 'skach', 'skewr', 'skrxp',
            'spcop', 'stlmt', 'thrsk', 'wkcti', 'wkna8', 'wknck', 'wkovl', 'wkpos', 'wtoeg',
        ],
        'scalar_cols': [],
    },
    190411: {
        'one_hot_cols': [],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
            'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
            'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
            'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48',
        ],
    },
    146820: {
        'one_hot_cols': [],
        'scalar_cols': [
            'GLCM_Pan', 'Mean_G', 'Mean_R', 'Mean_NIR', 'SD_Plan',
        ],
    },
    359974: {
        'one_hot_cols': [],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
        ],
    },
    359968: {
        'one_hot_cols': [
            'area_code', 'international_plan', 'voice_mail_plan', 'number_customer_service_calls',
        ],
        'scalar_cols': [
            'state', 'account_length', 'phone_number', 'number_vmail_messages', 'total_day_minutes',
            'total_day_calls', 'total_day_charge', 'total_eve_minutes', 'total_eve_calls',
            'total_eve_charge', 'total_night_minutes', 'total_night_calls', 'total_night_charge',
            'total_intl_minutes', 'total_intl_calls', 'total_intl_charge',
        ],
    },
    359975: {
        'one_hot_cols': [],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
            'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
            'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36',
        ],
    },
    359972: {
        'one_hot_cols': [],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
            'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        ],
    },
    168350: {
        'one_hot_cols': [],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5',
        ],
    },
    359969: {
        'one_hot_cols': [],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
            'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
            'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
            'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50',
            'V51',
        ],
    },
    359970: {
        'one_hot_cols': [],
        'scalar_cols': [
            'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14',
            'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26',
            'X27', 'X28', 'X29', 'X30', 'X31', 'X32',
        ],
    },
    359971: {
        'one_hot_cols': [
            'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
            'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
            'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
            'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
            'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
            'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
            'Statistical_report',
        ],
        'scalar_cols': [],
    },
    359979: {
        'one_hot_cols': [
            'RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 'ROLE_TITLE',
            'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE',
        ],
        'scalar_cols': [],
    },
    359980: {
        'one_hot_cols': [
            'V7', 'V8', 'V15', 'V16', 'V23', 'V24', 'V31', 'V32', 'V39', 'V40', 'V47', 'V48', 'V55',
            'V56', 'V63', 'V64', 'V71', 'V72', 'V79', 'V80', 'V87', 'V88', 'V92', 'V96', 'V100',
            'V104', 'V108', 'V112', 'V116',
        ],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V17',
            'V18', 'V19', 'V20', 'V21', 'V22', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V33',
            'V34', 'V35', 'V36', 'V37', 'V38', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V49',
            'V50', 'V51', 'V52', 'V53', 'V54', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V65',
            'V66', 'V67', 'V68', 'V69', 'V70', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V81',
            'V82', 'V83', 'V84', 'V85', 'V86', 'V89', 'V90', 'V91', 'V93', 'V94', 'V95', 'V97',
            'V98', 'V99', 'V101', 'V102', 'V103', 'V105', 'V106', 'V107', 'V109', 'V110', 'V111',
            'V113', 'V114', 'V115', 'V117', 'V118',
        ],
    },
    359992: {
        'one_hot_cols': [
            'ad_id', 'advertiser_id', 'keyword_id', 'title_id', 'description_id', 'user_id',
        ],
        'scalar_cols': [
            'impression', 'url_hash', 'depth', 'position', 'query_id',
        ],
    },
    359981: {
        'one_hot_cols': [],
        'scalar_cols': [
            'white_piece0_strength', 'white_piece0_file', 'white_piece0_rank',
            'black_piece0_strength', 'black_piece0_file', 'black_piece0_rank',
        ],
    },
    359982: {
        'one_hot_cols': [
            'V2', 'V3', 'V4', 'V5', 'V7', 'V8', 'V9', 'V11', 'V16',
        ],
        'scalar_cols': [
            'V1', 'V6', 'V10', 'V12',sddasfasdasfd    359987: {
        'one_hot_cols': [],
        'scalar_cols': [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', : {
        'one_hot_cols': [],
        'scalar_cols': [sasdfasdf
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
            'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
            'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
            'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50',
            'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62',
            'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74',
            'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86',
            'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98',
            'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109',
            'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120',
            'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131',
            'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141', 'V142',
            'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153',
            'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164',
            'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175',
            'V176', 'V177', 'V178', 'V179', 'V180',
        ],
    },
    359984: {
        'one_hot_cols': [],
        'scalar_cols': [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14',
            'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
            'V27',
        ],
    },
    359977: {
        'one_hot_cols': [
            'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'c1', 'c2',
            'c3', 'c4', 'c5', 'c6', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'e1', 'e2', 'e3', 'e4',
            'e5', 'e6', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6',
        ],
        'scalar_cols': [],
    },
}


def load_task_data(suite_directory: Path, task_id: int, y_label: str):
    """
    Load one OpenML suite dataset and its one-hot and numeric feature columns.

    The feature columns are read from TASK_FEATURE_COLUMNS. The lists must cover every feature column
    of the CSV exactly once; a mismatch raises instead of training on the wrong columns.
    """
    csv_path = suite_directory / f'task_{task_id}.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    if task_id not in TASK_FEATURE_COLUMNS:
        raise ValueError(f"No feature columns for task {task_id} in TASK_FEATURE_COLUMNS.")

    data = pd.read_csv(csv_path)

    # check if the provided y_label is valid
    if y_label not in data.columns:
        raise ValueError(f"Invalid y_label '{y_label}'.")

    # the suite excludes datasets with missing values, so any null means the data is not what we expect
    null_columns = data.columns[data.isnull().any()].tolist()
    if null_columns:
        raise ValueError(f"Columns with null values in {csv_path.name}: {null_columns}")

    one_hot_cols = list(TASK_FEATURE_COLUMNS[task_id]['one_hot_cols'])
    scalar_cols = list(TASK_FEATURE_COLUMNS[task_id]['scalar_cols'])

    # the two lists must split the CSV's feature columns exactly: nothing missing, nothing listed twice
    listed = one_hot_cols + scalar_cols
    features = [col for col in data.columns if col != y_label]
    missing = sorted(set(features) - set(listed))
    unknown = sorted(set(listed) - set(features))
    repeated = sorted({col for col in listed if listed.count(col) > 1})
    if missing or unknown or repeated:
        raise ValueError(f"TASK_FEATURE_COLUMNS[{task_id}] does not match {csv_path.name}: "
                         f"missing {missing}, not in the CSV {unknown}, repeated {repeated}.")

    # a non-numeric column in scalar_cols would reach the models as strings
    non_numeric = [col for col in scalar_cols if not pd.api.types.is_numeric_dtype(data[col])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns listed in scalar_cols for task {task_id}: {non_numeric}")

    print(f"Loaded task {task_id} from {csv_path} with shape {data.shape}: "
          f"{len(one_hot_cols)} one-hot column(s), {len(scalar_cols)} numeric column(s).", flush=True)

    return data, one_hot_cols, scalar_cols


def is_run_complete(output_directory: str) -> bool:
    """
    Check if a run is complete by verifying the existence of best_results.json.
    A complete run will have: {output_directory}/best_results.json
    """
    results_file = os.path.join(output_directory, "best_results.json")
    return os.path.isfile(results_file)

if __name__ == "__main__":
    # get configs for running EA
    parser = argparse.ArgumentParser(description="Run EA CASH (full-pipeline optimization) on every dataset in an OpenML suite folder")
    # random seed for reproducibility
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility.')
    # suite folder (needs tasks_summary.csv and task_<id>.csv for each task)
    parser.add_argument('--data_path', type=str, required=True, help='Folder of OpenML suite files, e.g. Experiments/Data-Preprocessing/OpenML_Suite_271_Classification_CSV.')
    # optional subset of tasks
    parser.add_argument('--task_ids', type=int, nargs='+', default=None, help='Only run these task IDs. Default: every task in tasks_summary.csv.')
    # train proportion
    parser.add_argument('--train_p', type=float, required=True, help='Proportion of dataset to use for training.')
    # output directory (one task_<id> subfolder per dataset)
    parser.add_argument('--output_directory', type=str, required=True, help='Parent directory for output files; each dataset is written to <output_directory>/task_<id>/.')
    # number of generations
    parser.add_argument('--gens', type=int, required=True, help='Number of generations.')
    # population size
    parser.add_argument('--pop_size', type=int, required=True, help='Population size.')
    # number of cores
    parser.add_argument('--cores', type=int, required=True, help='Number of CPU cores to use.')
    # mutation probability
    parser.add_argument('--mut_prob', type=float, required=True, help='Probability of mutating each hyperparameter.')
    # crossover probability
    parser.add_argument('--crossover_prob', type=float, required=True, help='Probability of applying crossover.')
    # component (structural) mutation probability
    parser.add_argument('--component_mut_prob', type=float, required=True, help="Probability a node's component is resampled (structural mutation).")
    # TPE offspring probability
    parser.add_argument('--tpe_offspring', type=float, required=True, help='Probability that an offspring is chosen by the TPE from num_offspring candidate siblings.')
    # tournament size
    parser.add_argument('--tournament_size', type=int, required=True, help='Tournament size for parent selection.')
    # number of offspring for TPE
    parser.add_argument('--num_offspring', type=int, required=True, help='Number of pseudo-offspring to generate for TPE.')
    # gamma for TPE
    parser.add_argument('--gamma', type=float, required=True, help='Gamma parameter for TPE (fraction of samples considered good).')
    # number of cross-validation folds
    parser.add_argument('--n_folds', type=int, default=10, help='Number of cross-validation folds. Default 10.')

    args = parser.parse_args()

    suite_directory = Path(args.data_path).expanduser().resolve()
    if not suite_directory.is_dir():
        parser.error(f"--data_path is not a folder: {suite_directory}")

    targets = load_suite_targets(suite_directory)
    if args.task_ids:
        known = {task_id for task_id, _ in targets}
        unknown = [task_id for task_id in args.task_ids if task_id not in known]
        if unknown:
            parser.error(f"task IDs not in {suite_directory / 'tasks_summary.csv'}: {unknown}")
        wanted = set(args.task_ids)
        targets = [(task_id, y_label) for task_id, y_label in targets if task_id in wanted]

    # print all argument values
    print("=" * 60, flush=True)
    print("CASH Experiment Configuration", flush=True)
    print("=" * 60, flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Data Path: {suite_directory}", flush=True)
    print(f"Tasks ({len(targets)}): {[task_id for task_id, _ in targets]}", flush=True)
    print(f"Train Proportion: {args.train_p}", flush=True)
    print(f"Output Directory: {args.output_directory}", flush=True)
    print(f"Generations: {args.gens}", flush=True)
    print(f"Population Size: {args.pop_size}", flush=True)
    print(f"Cores: {args.cores}", flush=True)
    print(f"Mutation Probability: {args.mut_prob}", flush=True)
    print(f"Crossover Probability: {args.crossover_prob}", flush=True)
    print(f"Component Mutation Probability: {args.component_mut_prob}", flush=True)
    print(f"TPE Offspring Probability: {args.tpe_offspring}", flush=True)
    print(f"Tournament Size: {args.tournament_size}", flush=True)
    print(f"Num Offspring: {args.num_offspring}", flush=True)
    print(f"Gamma: {args.gamma}", flush=True)
    print(f"CV Folds: {args.n_folds}", flush=True)
    print(f"Scalers: {SCALERS}", flush=True)
    print(f"Transformers: {TRANSFORMERS}", flush=True)
    print(f"Selectors: {SELECTORS}", flush=True)
    print(f"Predictors: {CLASSIFIERS}", flush=True)
    print("=" * 60, flush=True)
    print('', flush=True)

    # initialize ray
    if not ray.is_initialized():
        # Ray workers do not inherit this script's sys.path, so hand them the repo root via
        # PYTHONPATH; otherwise they cannot import Source when launched outside the repo root
        ray.init(num_cpus=args.cores, include_dashboard=False, ignore_reinit_error=True,
                 runtime_env={'env_vars': {'PYTHONPATH': REPO_ROOT}})
    print(f"Ray initialized with {args.cores} cores.", flush=True)

    # run one EA per dataset; a failing dataset is reported and the rest still run
    failed = []
    for task_id, y_label in targets:
        run_directory = os.path.join(args.output_directory, f"task_{task_id}")
        print("=" * 60, flush=True)
        print(f"Task {task_id} (target '{y_label}')", flush=True)

        # check if run is already complete
        if is_run_complete(run_directory):
            print(f"Run already complete. Results exist at: {run_directory}/best_results.json", flush=True)
            print("Skipping execution.", flush=True)
            continue

        try:
            data_set, one_hot_cols, scalar_cols = load_task_data(suite_directory, task_id, y_label)

            # create the CASH EA
            ea = optimizer.EA(
                seed=args.seed,
                pop_size=args.pop_size,
                cores=args.cores,
                mut_prob=args.mut_prob,
                tpe_offspring=args.tpe_offspring,
                tournament_size=args.tournament_size,
                num_offspring=args.num_offspring,
                crossover_prob=args.crossover_prob,
                scalers=SCALERS,
                transformers=TRANSFORMERS,
                selectors=SELECTORS,
                predictors=CLASSIFIERS,
                component_mut_prob=args.component_mut_prob,
                gamma=args.gamma,
            )
            print(f"CASH EA initialized", flush=True)

            # load data
            ea.load_data_pd(
                data=data_set,
                target_label=y_label,
                train_p=args.train_p,
                one_hot_cols=one_hot_cols,
                scalar_cols=scalar_cols,
                n_folds=args.n_folds
            )
            print(f"CASH EA data loaded", flush=True)

            # run evolution
            ea.evolve(gens=args.gens)

            # save results
            ea.save_results(save_dir=run_directory)
        except Exception:
            traceback.print_exc()
            failed.append(task_id)
            print(f"Task {task_id} failed; continuing with the next dataset.", flush=True)

    # shutdown ray
    ray.shutdown()

    print("=" * 60, flush=True)
    print(f"Finished {len(targets)} task(s); failed: {failed if failed else 'none'}", flush=True)
    if failed:
        sys.exit(1)
