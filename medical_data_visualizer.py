import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. استيراد البيانات من ملف CSV
df = pd.read_csv('medical_examination.csv')

# 2. إضافة عمود 'overweight' لحساب زيادة الوزن بناءً على BMI
# BMI = الوزن بالكيلوجرام / (الطول بالمتر)^2
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (bmi > 25).astype(int)

# 3. توحيد البيانات (0 دائماً جيد، و 1 دائماً سيء)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4. رسم المخطط التصنيفي (Categorical Plot)
def draw_cat_plot():
    # 5. إعادة تشكيل البيانات باستخدام pd.melt
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. تجميع البيانات وتجميع التكرارات
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. رسم المخطط باستخدام sns.catplot
    catplot = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar'
    )

    # 8. حفظ الشكل في متغير fig
    fig = catplot.fig

    # 9. لا تعدل السطرين التاليين
    fig.savefig('catplot.png')
    return fig


# 10. رسم الخريطة الحرارية (Heat Map)
def draw_heat_map():
    # 11. تنظيف البيانات واستبعاد القيم غير الصحيحة أو المتطرفة
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. حساب مصفوفة الارتباط (Correlation Matrix)
    corr = df_heat.corr()

    # 13. إنشاء القناع (Mask) للجزء العلوي من المثلث
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. إعداد شكل matplotlib
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. رسم الخريطة الحرارية باستخدام seaborn
    sns.heatmap(
        corr,
        annot=True,
        fmt='.1f',
        mask=mask,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.5},
        ax=ax
    )

    # 16. لا تعدل السطرين التاليين
    fig.savefig('heatmap.png')
    return fig