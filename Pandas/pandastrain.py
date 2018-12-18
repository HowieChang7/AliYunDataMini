# -*- coding: utf-8 -*-

# pandas.read_csv(filepath, sep=', ' ,header='infer', names=None)
#
# filepath:文本文件路径；sep:分隔符；header默认使用第一行作为列名，如果header=None则pandas为其分配默认的列名；也可使用names传入列表指定列名

#查看数据的信息，包括每个字段的名称、非空数量、字段的数据类型
# data.info()

#从data.info()得知，contbr_employer、contbr_occupation均有少量缺失值,均填充为NOT PROVIDED
# data['contbr_employer'].fillna('NOT PROVIDED',inplace=True)
# data['contbr_occupation'].fillna('NOT PROVIDED',inplace=True)

#选取候选人为Obama、Romney的子集数据
#data_vs = data[data['cand_nm'].isin(['Obama, Barack','Romney, Mitt'])].copy()