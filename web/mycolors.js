const CSS_COLOR_NAMES = [
    "Tomato",
    "Orange",
    "DodgerBlue",
    "Sienna",
    "Gray",
    "SlateBlue",
    "Violet",
    "LightGray",
    "Yellow",
    "Fuchsia",
    "Green",
    "Cyan",
    "Chartreuse",
    "OrangeRed",
    "Olive"
];

const ALL_LABELS = {
    'PAM50_and_Claudin-low_(CLOW)_Molecular_Subtype':
        {'LumA': 0, 'LumB': 1, 'Basal': 2, 'HER2E':3, 'normal':4},
    'CLS_HistoAnno':
        {'Lobular':0, 'Ductal':1, 'Other':2},
    'CLS_IHC_HER2':
        {'Negative':0,  'Positive':1, 'Other':2},

    'survival_groups':
        {'LowRisk':0, 'HighRisk':1},
    'Stage':
        {'StageI_II':0, 'StageM':1},

    'treatment_group':
        {'1 (control Ab)':1, '2 (Anti-TGFb)':2},
    'met_weight_label':
        {'group0':0, 'group1':1, 'group2':2},

    'TP53_cls':
        {'Gain_Or_Unknown_Or_NaN':0, 'Loss':1, 'other':2},

    'ResponseGroup':
        {'PD':0, 'SD':1, 'PR':2, 'CR':3},
    'response_groups':
        {'BadResponse': 0, 'GoodResponse': 1},
    'group_label':
        {'group0': 0, 'group1': 1}
};
