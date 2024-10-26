import ast
import glob
import csv
import os
import pickle
import numpy as np

# SD30_helium
# output = [{'timespan-[1501920252, 1504548540, 1]': []}, {'timespan-[1635896053, 1638524341, 2]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3148.3477'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2645.2021'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3391.5127'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4498.968'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3404.7593'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5191.675'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3845.1052'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3171.9802'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '4572.058']]}, {'timespan-[1638524621, 1641152909, 3004]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3061.8303'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2315.369'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2890.3982'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4685.2515'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3516.8513'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5450.52'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3237.503'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3191.6804'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2947.3167']]}, {'timespan-[1641154548, 1643782836, 3929]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2982.41'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2871.9197'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2932.4424'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '3968.1948'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2863.5564'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4934.7915'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3187.2993'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2191.6687'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2830.21']]}, {'timespan-[1643783404, 1646411692, 4324]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3198.0015'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '3129.7915'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3574.4487'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4634.913'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3012.5662'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5148.019'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3123.6848'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2818.1997'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3759.8516']]}, {'timespan-[1646412680, 1649040968, 3458]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2900.6377'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2939.4922'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3018.8022'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4217.2114'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3081.5376'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4663.212'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3421.3594'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2959.1687'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3749.4995']]}, {'timespan-[1649042174, 1651670462, 2514]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2953.9595'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2633.343'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3189.4375'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4409.0503'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2853.9453'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5065.988'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3472.2244'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2953.314'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3562.8762']]}, {'timespan-[1651671228, 1654299516, 1482]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3109.8518'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2512.5437'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3206.727'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4692.47'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3553.031'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5358.694'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3371.0317'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2554.4268'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '4498.555']]}, {'timespan-[1654301298, 1656929586, 1096]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3201.6216'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2586.846'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2659.0054'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4342.9473'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2593.4363'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3553.7517'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3095.4556'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2871.2053'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3140.569']]}, {'timespan-[1656946921, 1659575209, 1075]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3096.2495'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '3303.1782'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3720.4595'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4783.321'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2958.2476'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5589.598'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3655.8472'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2832.7935'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '4512.353']]}, {'timespan-[1659577116, 1662205404, 919]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3140.5862'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2955.2012'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '4181.419'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4306.379'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3880.524'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5718.4287'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3695.457'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3209.2393'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3565.361']]}, {'timespan-[1662212994, 1664841282, 1220]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3295.696'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2795.5117'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3407.1086'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4516.8955'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3018.7852'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '6075.65'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3426.1914'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2659.6365'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3976.929']]}, {'timespan-[1664841650, 1667469938, 1280]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3114.5906'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2979.2402'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3443.19'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4634.2695'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3617.5742'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5123.612'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3632.2688'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2471.0847'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3717.9956']]}, {'timespan-[1667477540, 1670105828, 1034]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3166.6367'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2647.123'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3204.629'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4291.315'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3171.0193'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5166.5737'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3313.906'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2532.8425'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3297.0261']]}, {'timespan-[1670112648, 1672740936, 1284]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3210.0'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '3340.5981'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2997.1921'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4023.0806'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3478.9565'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4652.4077'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3397.358'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2699.5806'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3173.7993']]}, {'timespan-[1672747521, 1675375809, 1459]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2809.819'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2073.4316'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2712.1982'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4058.6165'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3393.6099'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4470.3354'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3689.6902'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2830.101'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3628.509']]}, {'timespan-[1675375955, 1678004243, 1239]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3036.6934'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2673.0625'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2667.8801'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4672.7476'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3052.6108'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5414.1567'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3723.6094'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3293.3928'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3433.947']]}, {'timespan-[1678004317, 1680632605, 4608]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2898.807'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2531.3728'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3071.726'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '3782.2676'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3297.6624'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4647.9526'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3816.8904'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3187.3027'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3235.964']]}, {'timespan-[1680632608, 1683260896, 4559]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '3464.3494'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '3162.2454'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '3475.143'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4191.8374'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '6730.183'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '5026.0093'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3732.6533'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '3106.4985'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '3819.3877']]}]
# SEA30_helium
output = [{'timespan-[1635896268, 1638524556]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2604.4692'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1030.0905'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1255.3203'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '1667.6445'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2368.3657'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2017.2415'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2374.4421'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1056.4209'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1318.2003']]}, {'timespan-[1638525049, 1641153337]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2568.7869'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '934.3393'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1162.8083'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2500.169'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1177.0708'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1465.612']]}, {'timespan-[1641154285, 1643782573]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '1984.0078'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1211.2169'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1284.555'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '2823.754'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3092.4683'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2911.5425'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2417.308'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1259.3456'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1634.1616']]}, {'timespan-[1643782688, 1646410976]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2243.7197'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1245.4286'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1549.7991'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '2499.3372'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3171.8716'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3381.7092'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2704.6548'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1935.1259'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2207.2634']]}, {'timespan-[1646412372, 1649040660]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2487.181'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1143.9791'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1616.763'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '2717.722'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2885.149'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3370.9558'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2463.5867'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1697.067'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2106.498']]}, {'timespan-[1649041659, 1651669947]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2231.6682'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1377.3357'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1528.1565'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '2083.866'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2671.008'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2149.6785'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2730.6206'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1212.6995'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1672.2434']]}, {'timespan-[1651670093, 1654298381]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2474.6655'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1069.3596'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1122.98'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '2409.5686'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2994.0017'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3207.68'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2466.938'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1143.7452'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1500.6298']]}, {'timespan-[1654299421, 1656927709]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2429.9832'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1242.863'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1511.3627'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '2281.4714'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3008.55'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3139.0764'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2495.0088'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1102.6265'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1293.7515']]}, {'timespan-[1656928416, 1659556704]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2315.2664'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1155.8884'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1330.5139'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '3387.3765'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2774.0842'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3233.8882'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2730.0095'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1275.272'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1376.2615']]}, {'timespan-[1659558771, 1662187059]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2256.7173'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1379.244'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1455.4543'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '2381.4404'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '1720.401'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2162.1206'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2645.0146'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1406.4465'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1864.7751']]}, {'timespan-[1662188017, 1664816305]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2139.0884'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1005.4674'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1146.7332'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '2893.9512'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3490.4988'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3484.2815'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2710.7673'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1115.5552'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1359.1348']]}, {'timespan-[1664817247, 1667445535]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2388.3408'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1206.6405'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1449.9829'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2807.5122'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1290.8182'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1553.3966']]}, {'timespan-[1667445931, 1670074219]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2423.734'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1335.5262'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1272.499'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '2209.2222'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '1433.5565'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '1729.4646'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2515.7202'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1480.6077'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1568.2676']]}, {'timespan-[1670074662, 1672702950]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2527.7263'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1088.0005'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1162.1332'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '2709.9949'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3497.4722'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4152.147'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2418.784'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1437.2137'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1923.0874']]}, {'timespan-[1672703323, 1675331611]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2690.1243'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '734.4363'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1028.7834'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '2087.3506'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2325.1606'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '1942.6198'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2657.0215'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1258.7013'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1429.3624']]}, {'timespan-[1675332204, 1677960492]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2529.4575'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '928.48987'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1032.0994'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2426.2388'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1245.6614'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1805.3197']]}, {'timespan-[1677960546, 1680588834]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2367.5747'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1119.4985'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1262.8776'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4155.9883'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '4249.502'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '4011.8782'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '3082.653'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '958.059'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2577.5186']]}, {'timespan-[1680588975, 1683217263]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '2725.775'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1415.4572'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2036.613'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '2830.209'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1696.9727'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1678.135']]}]
# SF30_helium
# output = [{'timespan-[1501859456, 1504487744, 1]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '6007.826'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '1186.188'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '1150.261']]}, {'timespan-[1635890982, 1638519270, 13]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7664.6943'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2074.319'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2102.7546'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4337.6606'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2480.7537'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2648.4497'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7936.956'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1932.8818'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1978.0084']]}, {'timespan-[1638519276, 1641147564, 51203]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7510.135'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2185.816'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2269.6978'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4689.3965'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3296.69'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3610.1096'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '8123.4375'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2240.2036'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2378.7742']]}, {'timespan-[1641147650, 1643775938, 60709]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '6836.775'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2274.4893'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2260.7974'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4367.2065'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3009.8367'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3308.807'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7726.193'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2593.6992'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2633.5803']]}, {'timespan-[1641147650, 1643775938, 60709]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7586.0024'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2268.738'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2251.4219'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4226.4473'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2997.107'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3388.1987'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7497.978'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2696.6821'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2674.945']]}, {'timespan-[1641147650, 1643775938, 60709]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7701.6304'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2273.674'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2145.022'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4399.6343'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2981.9714'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3126.792'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7519.238'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2609.1133'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2703.4675']]}, {'timespan-[1643775940, 1646404228, 62572]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7590.315'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2046.9113'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2095.3562'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4414.7944'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3075.413'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3218.448'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7316.951'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2319.7122'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2352.1177']]}, {'timespan-[1643775940, 1646404228, 62572]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7902.884'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2042.1448'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2133.5574'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4802.3506'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '3075.9026'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3191.5554'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7985.3174'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2300.5583'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2428.6208']]}, {'timespan-[1646404229, 1649032517, 56950]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '8123.7026'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2311.8357'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2325.9226'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4770.3354'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2709.4475'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2982.136'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7741.288'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2315.5415'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2359.2976']]}, {'timespan-[1649032732, 1651661020, 42181]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7543.5454'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2083.7468'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2157.9172'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4522.353'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2510.0154'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2830.5378'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7676.3496'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2085.0437'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2179.0542']]}, {'timespan-[1651661105, 1654289393, 23423]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7502.5747'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2194.295'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2264.981'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4901.5376'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2843.1663'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3570.207'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7964.311'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2166.8943'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2259.4685']]}, {'timespan-[1654289759, 1656918047, 17242]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7281.8076'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2131.9172'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2292.802'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4423.437'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2531.468'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2981.6494'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '8191.4155'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1975.7416'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2083.9846']]}, {'timespan-[1654289759, 1656918047, 17242]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7493.5205'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2131.4387'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2195.0251'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '5231.3906'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2531.0757'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2646.099'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '8113.6396'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1974.5251'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2072.7708']]}, {'timespan-[1654289759, 1656918047, 17242]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7524.614'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2129.6704'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2209.207'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '5243.4087'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2535.504'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2743.1892'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7624.3623'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '1972.935'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '1991.05']]}, {'timespan-[1656918908, 1659547196, 16679]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '6836.802'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2310.7268'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2206.5735'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4560.984'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2709.3955'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2779.7427'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7960.789'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2071.4863'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2131.2466']]}, {'timespan-[1656918908, 1659547196, 16679]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '6894.234'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2121.2642'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2152.0032'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4849.08'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2685.8787'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2995.5515'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '8194.018'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2080.0112'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2097.183']]}, {'timespan-[1656918908, 1659547196, 16679]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7088.8525'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2126.06'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2161.0334'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4748.9897'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2686.0845'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2879.966'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7210.3335'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2094.595'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2186.8943']]}, {'timespan-[1659547633, 1662175921, 12623]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7522.202'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2125.8774'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2229.5493'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4235.9956'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2824.72'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '3197.524'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7633.822'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2353.2373'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2365.2893']]}, {'timespan-[1659547633, 1662175921, 12623]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7650.0737'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2105.7917'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2231.508'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4775.5146'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2838.4736'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2913.516'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7742.76'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2355.129'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2366.476']]}, {'timespan-[1659547633, 1662175921, 12623]': [[9, 'random', 0, 'CoMLoss()', '0.2testsize_test', '7679.4146'], [9, 'random', 0, 'MSELoss()', '0.2testsize_test', '2104.5415'], [9, 'random', 0, 'SlicedEarthMoversDistance()', '0.2testsize_test', '2187.527'], [9, 'grid2', 0, 'CoMLoss()', '0.2testsizegrid2_test', '4775.0137'], [9, 'grid2', 0, 'MSELoss()', '0.2testsizegrid2_test', '2825.8337'], [9, 'grid2', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid2_test', '2984.312'], [9, 'grid5', 0, 'CoMLoss()', '0.2testsizegrid5_test', '7877.6685'], [9, 'grid5', 0, 'MSELoss()', '0.2testsizegrid5_test', '2349.8484'], [9, 'grid5', 0, 'SlicedEarthMoversDistance()', '0.2testsizegrid5_test', '2395.558']]}]
# top 50 cities
input_file = "../2024_08_30_parsed.txt"
process_from_file=False

process_all_results_logs=True
results_pattern = r'results/*results.txt'
process_all_results_filename = '20240925_normal_1500cities_combined_results_rss.csv'
# this dir should contain ALL cities
generated_datasets_dir = r"C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240928_15000cities_denylist\generated"
# generated_datasets_dir = r"C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240921_15000cities_normal\generated"
# generated_datasets_dir = r"C:\Users\ps\OneDrive\Documents\DL_Image_Localization_Results\20240921_15000cities_normal_remainder1\generated"

def count_lines_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            line_count = sum(1 for _ in file)
        return line_count
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def process_all_results_logs(process_all_results_filename = process_all_results_filename):
    results_dict = {}
    for file_name in glob.glob(results_pattern):
        with open(file_name, 'r',encoding="ISO-8859-1") as file:
            # Some have more than one line per file due to filename timestamp convention
            # Read the file content and store it in the dictionary
            for line in file:
                count = 2
                if file_name in results_dict:
                    new_file_name = file_name + str(count)
                    count += 1
                else:
                    results_dict[file_name] = line
    bad_output_filename = process_all_results_filename.strip('.csv') + "_bad.csv"
    with (open(process_all_results_filename, 'w', newline='',encoding="ISO-8859-1") as outfile,
          open(bad_output_filename, 'w', newline='',encoding="ISO-8859-1") as bad_outfile):
        writer = csv.writer(outfile)
        writer_bad = csv.writer(bad_outfile)

        for file_name in results_dict:
            data_count = 0
            try:
                result = ast.literal_eval(results_dict[file_name])
                file_id = result["params"]["data_filename"].split('/')[-1]
                file_id.strip(',') # need to remove any commas within file_id name so .csv file will work
                #print(list(result["data"].values())[0])

                for span in result["data"]:
                    # print(results["data"])
                    # print(f"span {span}")
                    # print(f'results["data"][span] {results["data"][span]}')
                    # print(f"span.keys() = {span.keys()}")
                    min_error = 9999.0
                    # span_key = list(span.keys())[0]
                    # print(f"span[span_key] {span[span_key]}")
                    for row in result["data"][span]:
                        # print(result)
                        if float(row[-1]) < min_error:
                            min_error = float(row[-1])
                            #min_key = dataset_id
                    if min_error < 9999.0:
                        print(f"{file_name},{file_id},{min_error}")
                        writer.writerow([file_name,file_id,min_error])
                    else:
                        writer_bad.writerow([file_name, file_id, min_error])

                #print(result)


            except (SyntaxError, ValueError) as e:
                continue
                print(f"Error processing {file_name}: {e}")


def process_results_old_format():
    if process_from_file:
        with open(input_file,"r") as f:
            for line in f:
                results = ast.literal_eval(line)
                dataset_id = results["params"]["data_filename"]

                for span in results["data"]:
                    # print(results["data"])
                    # print(f"span {span}")
                    # print(f'results["data"][span] {results["data"][span]}')
                    #print(f"span.keys() = {span.keys()}")
                    min_error = 9999.0
                    # span_key = list(span.keys())[0]
                    # print(f"span[span_key] {span[span_key]}")
                    for result in results["data"][span]:
                        # print(result)
                        if float(result[-1]) < min_error:
                            min_error = float(result[-1])
                            min_key = dataset_id
                    print(f"{dataset_id} : {min_error}")
    else: # process from direct variable
        for span in output: # for timespan results
            # print(f"span {span}")
            # print(f"span.keys() = {span.keys()}")
            min_error = 9999.0
            span_key = list(span.keys())[0]
            # print(f"span[span_key] {span[span_key]}")
            for result in span[span_key]:
                # print(result)
                if float(result[-1]) < min_error:
                    min_error = float(result[-1])
                    min_key = span_key
            print(f"{span_key} : {min_error}")

def get_generated_datasets_city_list(datasets_dir=generated_datasets_dir):
    generated_city_datasets = []
    for file in os.listdir(datasets_dir):
        if file.endswith(".csv"):
            generated_city_datasets.append(file)
    return generated_city_datasets

def find_missing_results(results_file_path=process_all_results_filename):
    generated_city_list = get_generated_datasets_city_list()
    bad_output_filename = process_all_results_filename.strip('.csv') + "_bad.csv"


    cities_data = pickle.load(open('cities15000_dict_all.pickle', 'rb'))
    # cities_data = pickle.load(open('missing_city_dict_remainder1.pickle', 'rb'))


    missing_city_dict = {}

    with (open(results_file_path,"r",encoding="ISO-8859-1") as file,
          open(bad_output_filename,"r",encoding="ISO-8859-1") as bad_file):
        cities_with_results = []
        cities_with_bad_results = []
        for line in file:
            city_id = line.split(",")[1]
            cities_with_results.append(city_id)
        for line in bad_file:
            city_id = line.split(",")[1]
            cities_with_bad_results.append(city_id)



    for city_id in generated_city_list:
        if city_id not in cities_with_results and city_id not in cities_with_bad_results:
            geonameid = city_id.split("_")[0].strip('"')
            missing_city_dict[geonameid] = cities_data[geonameid]

    pickle.dump(missing_city_dict,open('20240928_missing_city_dict_remainder.pickle','wb'))

    missing_data = [x for x in generated_city_list if x not in cities_with_results]
    missing_data_bad = [x for x in generated_city_list if x not in cities_with_bad_results]
    missing_data_all = [x for x in generated_city_list if x not in cities_with_bad_results and x not in cities_with_results]

    print(f"original pickle list {len(cities_data)}, generated_datasets {len(generated_city_list)}, "
          f"cities with results {len(cities_with_results)}, cities with results bad {len(cities_with_bad_results)}, "
          f"cities_without_results {len(missing_data)}, without results bad {len(missing_data_bad)}, "
          f"cities without results and bad {len(missing_data_all)}, missing_city_dict {len(missing_city_dict)}")



    return

def combine_csv_files_to_dict(list_of_files):
    # creates a dictionary from multiple results files (in case repeated runs were needed to produce all results)
    results_dict = {} # {<geonameid> : <result>}
    dupe_count = 0
    for in_file in list_of_files:
        with open(in_file, "r", encoding="ISO-8859-1") as file:
            for line in file:
                print(line)
                results_file = line.split(",")[0].split('\\')[1]
                city_id = line.split(",")[1]
                geonameid = line.split(",")[1].split("_")[0].strip('"')
                error = float(line.split(",")[-1])
                # print(geonameid,error)
                if geonameid not in results_dict:
                    results_dict[geonameid] = {}
                    results_dict[geonameid]['error'] = error
                    results_dict[geonameid]['results_file'] = results_file
                    results_dict[geonameid]['city_id'] = city_id
                else:
                    print(f"{geonameid} already in dict. Skipping.")
                    dupe_count += 1
    print(f"dupe count {dupe_count}, dict size {len(results_dict)}")
    # print(results_dict)
    return results_dict

def get_data_counts_to_dict(datasets_dir=generated_datasets_dir):
# saves and returns structure like {'<geonameid>':<number of samples>}
    pickle_file_name = "all_geonameid_sample_counts.pickle"
    if os.path.exists(pickle_file_name):
        results_dict = pickle.load(open(pickle_file_name, "rb"))
        return results_dict

    results_dict = {}
    for file_name in os.listdir(datasets_dir):
        if file_name.endswith(".csv"):
            try:
                file_path = os.path.join(datasets_dir, file_name)
                with open(file_path, "r", encoding="ISO-8859-1") as file:
                    line_count = len(file.readlines())
            except Exception as e:
                print(f"{file_name} failed to load because {e}")
                continue
            geonameid = file_name.split("_")[0].strip('"')
            if geonameid not in results_dict:
                results_dict[geonameid] = line_count

    pickle.dump(results_dict, open(pickle_file_name, "wb"))
    return results_dict


def make_merged_output_csv(cities_error_dict):
    # retrieves data counts from "generated" directory
    # uses cities_error_dict to combined counts with errors to generated merged output
    cities_counts_dict = get_data_counts_to_dict()

    for geonameid in cities_error_dict:
        # print(f"{geonameid}: {cities_counts_dict[geonameid]}")
        cities_error_dict[geonameid]["sample_counts"] = cities_counts_dict[geonameid]
        # print(cities_error_dict[geonameid])


    # print(cities_counts_dict)
    # print(cities_error_dict)

    with open('cities_error_and_counts.csv', 'w', newline='',encoding="ISO-8859-1") as csvfile:
        fieldnames = ['results_file','city_id','error','sample_counts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for geonameid in cities_error_dict:
            writer.writerow(cities_error_dict[geonameid])

def compare_errors(file1='20240926_top60euro_denylist_combined_results.csv',
                   file2='20240926_top60euro_normal_combined_results.csv'):
    # compare error results for matching cities in file
    # used to compare training with RSS vs straight [1]
    dict1 = {}
    dict2 = {}
    with open(file1, "r", encoding="ISO-8859-1") as f1, open(file2, "r", encoding="ISO-8859-1") as f2:
        for line in f1:
            city_id = line.split(",")[1]
            try:
                error = float(line.split(",")[-1].strip()) # [-1] instead pf [2] in case a comma in name breaks the split
            except:
                print(f"couldn't parse {line} in position [-1]")
            geonameid = city_id.split("_")[0].strip('"')
            if geonameid not in dict1:
                dict1[geonameid] = error

        for line in f2:
            city_id = line.split(",")[1]
            try:
                error = float(line.split(",")[-1].strip())  # [-1] instead pf [2] in case a comma in name breaks the split
            except:
                print(f"couldn't parse {line} in position [-1]")
            geonameid = city_id.split("_")[0].strip('"')
            if geonameid not in dict2:
                dict2[geonameid] = error

    diff_list = []
    error_list1 = []
    error_list2 = []
    for geonameid in dict1:
        if geonameid in dict2:
            error1 = dict1[geonameid]
            error_list1.append(error1)
            error2 = dict2[geonameid]
            error_list2.append(error2)
            diff = error1 - error2
            diff_list.append(diff)

    print(diff_list)
    print(len(diff_list),np.mean(diff_list), np.mean(error_list1), np.mean(error_list2))
    # 2.1736558422375345 1098.3729622656392 1096.1993064234016
    # showed negligible difference between the two..
    # print(dict1)
    # print(dict2)
    return

def test():
    cities_data = pickle.load(open('cities15000_dict_all.pickle', 'rb'))
    european_country_codes = [
        "AL", "AD", "AM", "AT", "BY", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "GE", "DE", "GR", "HU", "IS", "IE", "IT", "XK", "LV", "LI", "LT", "LU", "MT", "MD", "MC",
        "ME", "NL", "MK", "NO", "PL", "PT", "RO", "SM", "RS", "SK", "SI", "ES", "SE", "CH", "UA",
        "GB", "VA"
    ]
    for city in cities_data:
        # if cities_data[city]['country'] == 'US':
        #     print(cities_data[city])
        if cities_data[city]['country'] in european_country_codes:
            print(cities_data[city])

def get_top_population_cities(country_list = ['US'], n = 50, make_pickle = False, pickle_name = ''):
    # returns list of geonameids corresponding to cities from country list with the top <n> population
    cities_data = pickle.load(open('cities15000_dict_all.pickle', 'rb'))
    # print(cities_data.items())


    # Create a list of tuples (population, geonameid)
    population_geonameid_pairs = [(int(cities_data[geonameid]['population']),
                                   geonameid,cities_data[geonameid]['name'].strip(','))
                                  for geonameid in cities_data if cities_data[geonameid]['country'] in country_list]


    # Sort the list in descending order by population
    population_geonameid_pairs.sort(reverse=True)

    print(f"sorted population: {population_geonameid_pairs[:n]}")

    # Extract the top n geonameids
    top_geonameids = [geonameid for _, geonameid, _ in population_geonameid_pairs[:n]]

    if make_pickle:
        cities_dict = pickle.load(open('cities15000_dict_all.pickle', 'rb'))

        new_dict = {key: value for key, value in cities_dict.items() if key in top_geonameids}
        print(f"len of old dict {len(cities_dict)} len of new dict {len(new_dict)}")
        pickle.dump(new_dict, open(pickle_name, 'wb'))

    # print(top_geonameids)
    return top_geonameids

def get_results_by_geonameid(target_list, source_file = '20240925_normal_error_vs_elev_stdev_exact.csv'):
    with open(source_file, "r", encoding="ISO-8859-1") as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        data = {row[0]: float(row[1]) for row in reader}
    all_cities = pickle.load(open('cities15000_dict_all.pickle', 'rb'))
    count = 0
    sorted_by_error_city_list = []
    for geonameid in target_list:
        try:
            # print(f"{geonameid}:{all_cities[geonameid]['name']},{data[geonameid]}")
            count += 1
            sorted_by_error_city_list.append([int(geonameid), all_cities[geonameid]['name'],float(data[geonameid])])
        except:
            # print(f"{geonameid}:{all_cities[geonameid]['name']} has no data")
            continue

    sorted_by_error_city_list.sort(key=lambda x: x[2])
    for row in sorted_by_error_city_list:
        print(row)

    print(count)
    # for city in all_cities:
    #     print(f"{city}: {all_cities[city]}")
    # print(target_list)


def  compare_denylist_to_normal(denylist_source='20240912_deny_list_merged_output.csv',
                                normal_source='cities_error_and_counts.csv',
                                geonameid_reference='cities15000_dict_all.pickle'):
    new_dict = {}
    dupe_count = 0

    cities_dict_all = pickle.load(open(geonameid_reference, 'rb'))
    for line in (open(denylist_source,'r',encoding="ISO-8859-1")):
        # print(line.strip())
        city=line.split(',')[1].split('__')[0]
        # print(city)
        for geonameid in cities_dict_all:
            if cities_dict_all[geonameid]['name'] == city:
                if geonameid in new_dict:
                    # print(f"{geonameid} is duplicated")
                    dupe_count += 1
                    new_dict[geonameid]['valid'] = 0 # 0 indicates duplication and invalid
                else:
                    new_dict[geonameid] = {}
                    new_dict[geonameid]['name'] = city
                    new_dict[geonameid]['denylist_error'] = float(line.split(',')[-2])
                    new_dict[geonameid]['denylist_sample_counts'] = int(line.split(',')[-1])
                    new_dict[geonameid]['valid'] = 1 # 1 indicates partial entry, not yet valid

    for line in (open(normal_source, 'r', encoding="ISO-8859-1")):
        geonameid=line.split(',')[1].split('_')[0]
        print(geonameid,geonameid in new_dict)
        if geonameid in new_dict and new_dict[geonameid]['valid'] == 1:
            new_dict[geonameid]['valid'] = 2
            new_dict[geonameid]['normal_error'] = float(line.split(',')[-2])
            new_dict[geonameid]['normal_sample_counts'] = int(line.split(',')[-1])


    valid_entries = [new_dict[geonameid] for geonameid in new_dict if new_dict[geonameid]['valid'] == 2]
    print(valid_entries,len(valid_entries))

    return new_dict

if __name__ == '__main__':
    list_of_files = ['20240921_normal_1500cities_combined_results.csv',
                     '20240921_normal_1500cities_combined_results_remainder1.csv',
                     '20240921_normal_1500cities_combined_results_remainder2.csv']
    # test()
    # compare_denylist_to_normal()
    # # get_results_by_geonameid(get_top_population_cities()) # default top 50 US cities
    # # get_results_by_geonameid(get_top_population_cities([
    # #     "AL", "AD", "AM", "AT", "BY", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    # #     "GE", "DE", "GR", "HU", "IS", "IE", "IT", "XK", "LV", "LI", "LT", "LU", "MT", "MD", "MC",
    # #     "ME", "NL", "MK", "NO", "PL", "PT", "RO", "SM", "RS", "SK", "SI", "ES", "SE", "CH", "UA",
    # #     "GB", "VA"
    # # ],55))
    # get_top_population_cities(country_list=['US'], n=60, make_pickle=True, pickle_name = 'top_60_US_cities.pickle')
    #
    # get_top_population_cities(country_list=    [
    #         "AL", "AD", "AM", "AT", "BY", "BE", "BA", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    #         "GE", "DE", "GR", "HU", "IS", "IE", "IT", "XK", "LV", "LI", "LT", "LU", "MT", "MD", "MC",
    #         "ME", "NL", "MK", "NO", "PL", "PT", "RO", "SM", "RS", "SK", "SI", "ES", "SE", "CH", "UA",
    #         "GB", "VA"
    # ], n=60, make_pickle=True, pickle_name = 'top_60_Euro_cities.pickle')

    # compare_errors('20240927_top60US_denylist_combined_results2.csv','20240926_top60US_normal_combined_results.csv',)
    # compare_errors()

    # process_results_old_format()

    # combine_csv_files_to_dict(list_of_files)
    # get_data_counts_to_dict()
    # make_merged_output_csv(combine_csv_files_to_dict(list_of_files))
    # process_all_results_logs('20240928_1500cities_denylist_combined_results.csv')
    find_missing_results('20240928_1500cities_denylist_combined_results.csv')
    #print(get_generated_datasets_city_list())
    # process_results_old_format()
