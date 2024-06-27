import pandas as pd
#合并数据框
# 读取CSV文件
kehuxiadan = pd.read_csv('project-2024-a/data/ERP_KHXD.csv')    #客户下单信息表 
fahuojilu = pd.read_csv('project-2024-a/data/ERP_FHJL.csv')   #发货记录表 
yonghu = pd.read_csv('project-2024-a/data/sys_user.csv')   #系统用户表 
kehu = pd.read_csv('project-2024-a/data/ERP_KHXX.csv')   #客户信息表 
zidian = pd.read_csv('project-2024-a/data/sys_dict_data.csv')  #字典
cheliang = pd.read_csv('project-2024-a/data/ERP_CLXX.csv')     #车辆信息表 
fahuodixinxi = pd.read_csv('project-2024-a/data/ERP_FHDXX.csv')   #发货地信息表 
zhandianxinxi = pd.read_csv('project-2024-a/data/ERP_ZDXX.csv')    #客户站点信息表

# 关联数据
# 1. 关联发货记录表和客户下单信息表
merged_df = pd.merge(fahuojilu, kehuxiadan, left_on='khxd_id', right_on='id', suffixes=('_fhjl', '_khxd'))#on 列名   使用 khxd_id（发货记录中的客户下单 ID）和 id（客户下单信息中的 ID）进行匹配
# id 	发货记录 ID 
# ddap_id 	调度安排计划 ID 
# khxx_id 	车辆信息 ID 
# zdxx_id 	客户站点信息 ID 
# khxd_id 	客户下单 ID 
# fhdxx_id 	发货地信息 ID 
# hplx 	货品类型 ID 
# cpgg 	产品规格字典值 
# cppp 	产品品牌字典值 
# jz 	发货吨位 
# dzdw 	到站吨位 
# dj 	货物单价 
# hk 	货款（单价 x 净重） 
# clxx_id 	车辆信息 ID 
# create_time 	创建时间。（只有客户收货后才会创建这个记录） 

# id 	客户下单条目 ID 
# khxx_id 	客户信息 ID 
# zd_id 	站点信息 ID 
# hplx 	货品类型字典值，对应字典类型 dict_typp='ERP_HPLX' 
# cpgg 	产品规格字典值，对应字典类型 dict_typp='ERP_CPGG' 
# cppp 	产品品牌字典值，对应字典类型 dict_typp='ERP_CPPP' 
# fhdw 	需求吨位 
# hyfs 	货运方式字典值，对应字典类型 dict_typp='ERP_HYFS' 
# ywlx_code 	业务类型描述 
# fhd_id 	发货地信息 ID 
# dzsj 	到站时间 
# create_by 	创建该订单的我司销售人员（因为所有客户的订单都是由我司销售人员代下的。） 


# 2. 关联销售员信息
merged_df = pd.merge(merged_df, yonghu, left_on='create_by', right_on='user_name', how='left')  #  创建了的订单使用 create_by（订单创建者）和 user_name（用户表中的用户名）进行匹配

# 3. 关联客户信息
merged_df = pd.merge(merged_df, kehu, left_on='khxx_id_fhjl', right_on='id', how='left', suffixes=('', '_khxx')) #客户信息 使用 khxx_id_fhjl（发货记录中的客户信息 ID）和 id（客户信息表中的 ID）进行匹

# 4. 关联发货地信息
merged_df = pd.merge(merged_df, fahuodixinxi, left_on='fhdxx_id', right_on='id', how='left', suffixes=('', '_fhdxx'))#发货地信息id 	fhdxx_id（发货记录中的发货地信息 ID）和 id（发货地信息表中的 ID）进行匹配。
#发货记录 ID 
# ddap_id 	调度安排计划 ID 
# khxx_id 	车辆信息 ID 
# zdxx_id 	客户站点信息 ID 
# khxd_id 	客户下单 ID 
# fhdxx_id 	发货地信息 ID 
# hplx 	货品类型 ID 
# cpgg 	产品规格字典值 
# cppp 	产品品牌字典值 
# jz 	发货吨位 
# dzdw 	到站吨位 
# dj 	货物单价 
# hk 	货款（单价 x 净重） 
# clxx_id 	车辆信息 ID 
# create_time 	创建时间。（只有客户收货后才会创建这个记录） 

# 5. 关联站点信息
merged_df = pd.merge(merged_df, zhandianxinxi, left_on='zdxx_id', right_on='id', how='left', suffixes=('', '_zdxx'))#使用 zdxx_id（发货记录中的站点信息 ID）和 id（站点信息表中的 ID）进行匹配。

# 6. 关联车辆信息
merged_df = pd.merge(merged_df, cheliang, left_on='clxx_id', right_on='id', how='left', suffixes=('', '_clxx'))  #使用 clxx_id（发货记录中的车辆信息 ID）和 id（车辆信息表中的 ID）进行匹配。

# 7. 关联字典表信息
# 确保列类型一致 merged_df 中的 hplx_fhjl、cpgg_fhjl、cppp_fhjl 列和 zidian 表中的 dict_value 列全部转化为str
merged_df['hplx_fhjl'] = merged_df['hplx_fhjl'].astype(str)
merged_df['cpgg_fhjl'] = merged_df['cpgg_fhjl'].astype(str)
merged_df['cppp_fhjl'] = merged_df['cppp_fhjl'].astype(str)
zidian['dict_value'] = zidian['dict_value'].astype(str)

# 货品类型 使用 hplx_fhjl（发货记录中的货品类型 ID）和 dict_value（字典表中的值）进行匹配，且字典类型为 ERP_HPLX 
merged_df = pd.merge(merged_df, zidian[zidian['dict_type'] == 'ERP_HPLX'], left_on='hplx_fhjl', right_on='dict_value', how='left', suffixes=('', '_hplx'))
merged_df.rename(columns={'dict_label': 'dict_label_hplx'}, inplace=True)

# 产品规格 使用 cpgg_fhjl（发货记录中的产品规格 ID）和 dict_value（字典表中的值）进行匹配，且字典类型为 ERP_CPGG
merged_df = pd.merge(merged_df, zidian[zidian['dict_type'] == 'ERP_CPGG'], left_on='cpgg_fhjl', right_on='dict_value', how='left', suffixes=('', '_cpgg'))
merged_df.rename(columns={'dict_label': 'dict_label_cpgg'}, inplace=True)

# 产品品牌使用 cppp_fhjl（发货记录中的产品品牌 ID）和 dict_value（字典表中的值）进行匹配，且字典类型为 ERP_CPPP
merged_df = pd.merge(merged_df, zidian[zidian['dict_type'] == 'ERP_CPPP'], left_on='cppp_fhjl', right_on='dict_value', how='left', suffixes=('', '_cppp'))
merged_df.rename(columns={'dict_label': 'dict_label_cppp'}, inplace=True)

# 检查合并后的列名
print(merged_df.columns)

# 提取并重命名所需字段
final_df = merged_df[['id_fhjl', 'create_time', 'user_name', 'ywlx_code', 'khmc', 'mc', 'zdmc', 'dict_label_hplx', 'dict_label_cpgg', 'dict_label_cppp', 'jz', 'dzdw', 'dj', 'hk', 'cph']]
final_df.columns = ['fhjl_id', 'fhjl_time', 'sales_name', 'ywlx_code', 'khmc', 'fhdmc', 'zdmc', 'hplx', 'cpgg', 'cppp', 'fhdw', 'dzdw', 'dj', 'hk', 'cph']

# 保存到CSV文件
final_df.to_csv('project-2024-a/data/final_output.csv', index=False)
#fhjl_id:

# 来源: fahuojilu 表中的 id_fhjl
# 含义: 发货记录 ID
# fhjl_time:

# 来源: fahuojilu 表中的 create_time
# 含义: 发货记录创建时间
# sales_name:

# 来源: yonghu 表中的 user_name
# 含义: 创建订单的销售人员姓名
# ywlx_code:

# 来源: kehuxiadan 表中的 ywlx_code
# 含义: 业务类型描述
# khmc:

# 来源: kehu 表中的 khmc
# 含义: 客户名称
# fhdmc:

# 来源: fahuodixinxi 表中的 mc
# 含义: 发货地名称
# zdmc:

# 来源: zhandianxinxi 表中的 zdmc
# 含义: 站点名称
# hplx:

# 来源: zidian 表中的 dict_label（通过 dict_type 为 ERP_HPLX 过滤）
# 含义: 货品类型描述
# cpgg:

# 来源: zidian 表中的 dict_label（通过 dict_type 为 ERP_CPGG 过滤）
# 含义: 产品规格描述
# cppp:

# 来源: zidian 表中的 dict_label（通过 dict_type 为 ERP_CPPP 过滤）
# 含义: 产品品牌描述
# fhdw:

# 来源: fahuojilu 表中的 jz
# 含义: 发货吨位
# dzdw:

# 来源: fahuojilu 表中的 dzdw
# 含义: 到站吨位
# dj:

# 来源: fahuojilu 表中的 dj
# 含义: 货物单价
# hk:

# 来源: fahuojilu 表中的 hk
# 含义: 货款（单价 x 净重）
# cph:

# 来源: cheliang 表中的 cph
# 含义: 车牌号