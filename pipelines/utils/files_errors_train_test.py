import rasterio
import numpy as np
import os
from glob import glob
import json


def get_extensions(prefix):
    if prefix in {"S2", "gt", "PERMANENTWATERJRC"}:
        return [".tif"]
    if prefix in {"S2rgb", "maskrgb"}:
        return [".jpg"]
    if prefix == "meta":
        return [".json"]

    if prefix == "floodmaps":
        return [".cpg", ".prj", ".shx", ".dbf", ".shp"]

    raise NotImplementedError(f"Prefix {prefix} unknown")


def worldfloods_files(data_version="2", worldfloods_root="../WORLDFLOODS"):
    """ Join all the meta.json in tiffimages/meta, check all products exist and add split subset """
    out = []
    from worldfloods.vector_data import bbox_2_pol
    filenames_errors = FILENAMES_ERRORS_V1 if data_version == "1" else FILENAMES_ERRORS_V2

    pols_test = []
    codes_test = []
    for fn in TEST_FILENAMES:
        codes_test.append(fn.split("_")[0])
        s2_file = os.path.join(worldfloods_root, "tiffimages/S2", fn+".tif")
        with rasterio.open(s2_file) as s2_src:
            pols_test.append(bbox_2_pol(s2_src.bounds, shapelypolygon=True))

    # Code 9284 is the same as 284
    codes_test.append("EMSR284")
    codes_test = set(codes_test)

    files_s2 = glob(os.path.join(worldfloods_root, "tiffimages/S2/*.tif"))

    for f in sorted(files_s2):
        prod_id = os.path.splitext(os.path.basename(f))[0]

        # Obtain bounding box S2 file
        s2_file = os.path.join(worldfloods_root, "tiffimages/S2", prod_id + ".tif")
        with rasterio.open(s2_file) as s2_src:
            bounds = s2_src.bounds
        pol_bounds = bbox_2_pol(bounds, shapelypolygon=True)

        # Obtain metadata from tiffimages/meta
        meta_file = os.path.join(worldfloods_root, "tiffimages/meta", prod_id + ".json")
        if not os.path.exists(meta_file):
            print("Missing meta file for %s" % meta_file)
            continue
        with open(meta_file, "r") as fh:
            meta = json.load(fh)

        total_pixels = np.prod(meta["shape"])
        frac_invalids = meta["pixels invalid S2"] / total_pixels

        # Determine which subset the image belongs to
        if prod_id in TEST_FILENAMES:
            subset = "test"
        elif prod_id in VAL_FILENAMES:
            subset = "val"
        elif prod_id in filenames_errors:
            subset = "excluded (errors)"
        elif (data_version == "1") and (meta["source"] == "unosat"):
            subset = "excluded (unosat)"
        elif prod_id.startswith("EMSR") and any((code in prod_id for code in codes_test)):
            subset = "excluded (test overlap)"
        elif any((pol_bounds.intersects(p) for p in pols_test)):
            print("File %s excluded because overlaps test set" % os.path.basename(f))
            subset = "excluded (test overlap)"
        elif frac_invalids > .5:
            # print("File %s excluded because it has too many invalids" % os.path.basename(f))
            subset = "excluded (invalids)"
        else:
            subset = "train"

        # Check all generated products exist skip otherwise
        missing_files = False
        exception = ""
        for parent_folder, p in zip(["tiffimages", "tiffimages", "rgbimages", "rgbimages"],
                                    ["gt", "floodmaps", "S2rgb", "maskrgb"]):
            for c_ext in get_extensions(p):
                p_file = os.path.join(worldfloods_root, parent_folder, p, prod_id + c_ext)
                if not os.path.exists(p_file):
                    if subset not in {"train", "test", "val"}:
                        print(f"Subset {subset} Missing {p} product for file {prod_id}: {p_file}. skip")
                    else:
                        exception += f"Missing file in subset {subset}. Product missing: {p} for file {prod_id}: {p_file}. ABORT. Add to FILENAMEERRORS list!\n"
                    missing_files = True

        if exception != "":
            raise ValueError(exception)
        if missing_files:
            continue

        meta["subset"] = subset

        out.append({"id": prod_id, "meta": meta})

    return out


TEST_FILENAMES = ['EMSR286_08ITUANGONORTH_DEL_MONIT02_v1_observed_event_a',
                  'EMSR286_09ITUANGOSOUTH_DEL_MONIT02_v1_observed_event_a',
                  'EMSR333_01RATTALORO_DEL_MONIT01_v1_observed_event_a',
                  'EMSR333_02PORTOPALO_DEL_MONIT01_v1_observed_event_a',
                  'EMSR333_13TORRECOLONNASPERONE_DEL_MONIT01_v2_observed_event_a',
                  'EMSR342_06NORTHNORMANTON_DEL_v1_observed_event_a',
                  'EMSR342_07SOUTHNORMANTON_DEL_MONIT03_v2_observed_event_a',
                  'EMSR347_06MWANZA_DEL_v1_observed_event_a',
                  'EMSR347_07ZOMBA_DEL_MONIT01_v1_observed_event_a',
                  'EMSR347_07ZOMBA_DEL_v2_observed_event_a',
                  'EMSR9284_01YLITORNIONORTHERN_DEL_MONIT01_v1_observed_event_a']


FILES_EXCLUDED_BEFORE = [
    "10132016_Ashley_River_near_North_Charleston_SC", # Maybe good!
    "10132016_Cooper_River_at_Monks_Corner_SC", # Maybe good!
    "EMSR258_06VLORE_DEL_v2_observed_event_a", # Maybe good!
    "EMSR264_12FARAFANGANA_DEL_v2_observed_event_a", # Maybe good!
    "EMSR273_04GOMSIQE_DEL_MONIT04_v1_observed_event_a", # Maybe good!
    "EMSR274_01AMBILOBE_DEL_v2_observed_event_a", # Maybe good!
    "EMSR274_12ANTALAHA_DEL_v1_observed_event_a", # Maybe good!
    "EMSR274_14MAHAJANGA_DEL_v1_observed_event_a", # Maybe good!
    "EMSR311_05JACKSONVILLE_DEL_MONIT03_v1_observed_event_a", # Maybe good!
    "EMSR312_01LAOAG_GRA_v1_observed_event_a", # Maybe good!
    "EMSR312_07VIGAN_DEL_MONIT01_v1_observed_event_a", # Maybe good!
    "EMSR312_07VIGAN_DEL_v1_observed_event_a", # Maybe good!
    "EMSR312_08CANDON_DEL_MONIT01_v1_observed_event_a", # Maybe good!
    "EMSR312_08CANDON_DEL_v1_observed_event_a",# Maybe good!
    "EMSR329_05CASTIADAS_GRA_v3_observed_event_a",# Maybe good!
    "EMSR333_12BAGHERIA_DEL_v1_observed_event_a",# Maybe good!
    "EMSR339_05BEIRASE_DEL_v1_observed_event_a",# Maybe good!
    "EMSR346_03BEIRA_DEL_v1_observed_event_a", # Maybe good!
    "ST_20170426_WaterExtent_LesCayes_Cavaillon", #Maybe good!
    "TSX_20170308_WaterExtent_Maroantsetra", #Maybe good!
]


FILENAMES_ERRORS_V2 = [
    # Excluded GloFIMR
    "05042017_Castor_River_at_Zalma_MO0000000000-0000000000",
    "09262016_Cedar_River_at_Vinton_IA0000000000-0000000000",
    "09262016_Cedar_River_at_Vinton_IA0000000000-0000012544",
    "09262016_Cedar_River_at_Vinton_IA0000012544-0000000000",
    "09262016_Cedar_River_at_Vinton_IA0000012544-0000012544",
    "10132016_Lumber_River_at_Lumberton_NC",
    "09262016_Iowa_River_at_Belle_Plaine_IA0000000000-0000000000",
    "09262016_Iowa_River_at_Belle_Plaine_IA0000000000-0000012544",
    "09262016_Mississippi_River_at_Clinton_IA0000000000-0000000000",
    "09262016_Wapsipinicon_River_at_Independence_IA0000000000-0000000000",
    "09262016_Mississippi_River_at_Clinton_IA0000012544-0000000000",
    "09262016_Wapsipinicon_River_at_Independence_IA0000000000-0000012544",
    "09262016_Wapsipinicon_River_at_Independence_IA0000012544-0000000000",
    "10132016_Lumber_River_at_Lumberton_NC",
    # Excluded Copernicus EMS
    "EMSR249_03GALWAY_DEL_v2_observed_event_a",
    "EMSR260_03SORBOLO_GRA_v3_observed_event_a",
    "EMSR260_04SANTILARIODENZA_GRA_v3_observed_event_a",
    "EMSR260_05MONTECCHIOEMILIA_GRA_v3_observed_event_a",
    "EMSR260_06SANPOLODENZA_DEL_MONIT01_v3_observed_event_a",
    "EMSR279_05ZARAGOZA_DEL_v2_observed_event_a", # repeated bad
    "EMSR260_06SANPOLODENZA_DEL_v2_observed_event_a",
    "EMSR265_08FONTAINEBLEAU_DEL_v2_observed_event_a",
    "EMSR265_08FONTAINEBLEAU_DEL_MONIT02_v3_observed_event_a",
    "EMSR265_08FONTAINEBLEAU_DEL_MONIT01_v2_observed_event_a",
    "EMSR265_05NOISYLEGRAND_DEL_MONIT01_v1_observed_event_a",
    "EMSR265_04ESBLY_DEL_MONIT05_v1_observed_event_a",
    "EMSR265_01DRAVEIL_DEL_v1_observed_event_a",
    "EMSR265_01DRAVEIL_DEL_MONIT02_v1_observed_event_a",
    "EMSR265_01DRAVEIL_DEL_MONIT01_v1_observed_event_a",
    "EMSR265_12BONNARD_DEL_v1_observed_event_a",
    "EMSR265_13AUXERRE_DEL_MONIT01_v1_observed_event_a",
    "EMSR265_13AUXERRE_DEL_v1_observed_event_a",
    "EMSR267_01RUSNE_DEL_v1_observed_event_a",
    "EMSR268_01DAUGAVA_DEL_v2_observed_event_a",
    "EMSR268_02LIELUPE_DEL_v2_observed_event_a",
    "EMSR265_09VOULX_DEL_MONIT05_v2_observed_event_a",
    "EMSR260_07CAMPOGALLIANO_DEL_v1_observed_event_a",
    "EMSR264_14VANGAINDRANO_DEL_v2_observed_event_a",
    "EMSR264_18MIANDRIVAZODETAIL_DEL_v2_observed_event_a",
    "EMSR265_04ESBLY_DEL_MONIT06_v2_observed_event_a",
    "EMSR277_02DIDIMOTICHO_DEL_MONIT01_v2_observed_event_a",
    "EMSR277_02DIDIMOTICHO_DEL_v2_observed_event_a",
    "EMSR277_03FERES_DEL_MONIT01_v1_observed_event_a",
    "EMSR277_03FERES_DEL_v1_observed_event_a",
    "EMSR284_03KARUNKI_DEL_v1_observed_event_a",
    "EMSR286_07NECHI_DEL_v2_observed_event_a",
    "EMSR280_05MORA_DEL_v2_observed_event_a",
    "EMSR283_03ELVERUM_DEL_MONIT01_v1_observed_event_a",
    "EMSR283_03ELVERUM_DEL_MONIT02_v1_observed_event_a",
    "EMSR284_04TORNIONORTHERN_DEL_MONIT01_v1_observed_event_a",
    "EMSR284_04TORNIONORTHERN_DEL_MONIT02_v1_observed_event_a",
    "EMSR284_04TORNIONORTHERN_DEL_v1_observed_event_a",
    "EMSR311_08FAYETTEVILLE_DEL_MONIT01_v1_observed_event_a",
    "EMSR314_01PATEGI_DEL_v1_observed_event_a",
    "EMSR314_03ENUGUOTU_DEL_v1_observed_event_a",
    "EMSR314_04OTUOCHAAGULERI_DEL_v1_observed_event_a",
    "EMSR314_05ONITSHA_DEL_v1_observed_event_a",
    "EMSR323_07SANTLLORENCOVERVIEW_GRA_v1_observed_event_a",
    "EMSR323_02SONCARRIO_GRA_v1_observed_event_a",
    "EMSR323_08MANACOR_GRA_v1_observed_event_a",
    "EMSR323_09CALAMILLOR_GRA_v1_observed_event_a",
    "EMSR324_01CARCASSONNE_DEL_MONIT01_v1_observed_event_a",
    "EMSR324_01CARCASSONNE_DEL_MONIT02_v1_observed_event_a",
    "EMSR324_01CARCASSONNE_DEL_v1_observed_event_a",
    "EMSR311_03MYRTLEBEACH_DEL_MONIT01_v1_observed_event_a", # Almost!
    "EMSR311_04WILMINGTON_DEL_MONIT01_v1_observed_event_a", # Almost!
    "EMSR322_17MARIANNA_DEL_v1_observed_event_a",
    "EMSR332_04BASSANODELGRAPPA_DEL_MONIT01_v1_observed_event_a", # Clouds bad
    "EMSR332_04BASSANODELGRAPPA_DEL_v1_observed_event_a",
    "EMSR332_08PORDENONE_DEL_MONIT01_v1_observed_event_a",
    "EMSR332_08PORDENONE_DEL_v1_observed_event_a",
    "EMSR332_09VENEZIA_DEL_MONIT01_v1_observed_event_a",
    "EMSR348_08TICA_DEL_MONIT01_v1_observed_event_a", # Bad clouds
    "EMSR348_08TICA_DEL_MONIT02_v3_observed_event_a", # Bad clouds
    "EMSR348_08TICA_DEL_MONIT03_v2_observed_event_a", # Bad clouds
    "EMSR348_08TICA_DEL_v1_observed_event_a", # Bad clouds
    # Excluded UNOSAT
    "ST_20170812_WaterExtent_CentralBangladesh0000000000-0000025088",
    "ST_20170812_WaterExtent_CentralBangladesh0000012544-0000000000",
    "ST_20170812_WaterExtent_CentralBangladesh0000012544-0000012544",
    "ST_20170822_WaterExtent_CentralSouthernBangladesh0000000000-0000000000",
    "ST_20170822_WaterExtent_CentralSouthernBangladesh0000000000-0000012544",
    "ST_20170822_WaterExtent_CentralSouthernBangladesh0000025088-0000025088",
    "ST_20171009_WaterExtent_Sayaxche0000000000-0000000000",
    "ST_20171009_WaterExtent_Sayaxche0000000000-0000012544",
    "ST_20171009_WaterExtent_Sayaxche0000012544-0000000000",
    "ST_20180508_WaterExtent_CumulativeKenya0000000000-0000000000",
    "ST_20180508_WaterExtent_CumulativeKenya0000000000-0000012544",
    "ST_20180508_WaterExtent_CumulativeKenya0000000000-0000025088",
    "ST_20180508_WaterExtent_CumulativeKenya0000000000-0000037632",
    "ST_20180508_WaterExtent_CumulativeKenya0000012544-0000012544",
    "ST_20180508_WaterExtent_CumulativeKenya0000012544-0000025088",
    "ST_20180508_WaterExtent_CumulativeKenya0000012544-0000037632",
    "ST_20180508_WaterExtent_CumulativeKenya0000025088-0000000000",
    "ST_20170326_WaterExtent_OmusatiAndOshana0000012544-0000000000",
    "ST_20180508_WaterExtent_CumulativeKenya0000025088-0000025088",
    "ST_20180508_WaterExtent_CumulativeKenya0000050176-0000025088",
    "ST1_20161218_WaterExtent_ThuaThienHue0000012544-0000012544", # too much clouds
    "ST1_20161218_WaterExtent_ThuaThienHue0000012544-0000000000",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000037632-0000025088",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000037632-0000037632",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000037632-0000012544",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000037632-0000000000",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000012544-0000000000",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000000000-0000037632",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000000000-0000025088",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000000000-0000012544",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000000000-0000000000",
    "ST20170314_WaterExtent_OmusatiAndOshana0000000000-0000000000",
    "ST20170314_WaterExtent_OmusatiAndOshana0000012544-0000000000",
    "ST20170314_WaterExtent_OmusatiAndOshana0000012544-0000012544",
    "ST20180501_WaterExtent_MiddleJuba0000000000-0000000000",
    "ST20180501_WaterExtent_MiddleJuba0000000000-0000012544",
    "ST20180501_WaterExtent_MiddleJuba0000012544-0000000000",
    "ST20180501_WaterExtent_Somali0000000000-0000000000",
    "ST_20170812_WaterExtent_CentralBangladesh0000000000-0000000000",
    "ST20170104_WaterExtent_Narathiwat0000000000-0000000000",
    "ST20160724_WaterExtent_Dhaka_Rajshahi0000012544-0000012544",
    "ST20160724_WaterExtent_Dhaka_Rajshahi0000012544-0000000000",
    "ST20180501_WaterExtent_Somali0000000000-0000012544",
    "ST20180615_WaterExtent_Assam_Sylhet0000000000-0000000000",
    "ST20180615_WaterExtent_Assam_Sylhet0000000000-0000012544",
    "ST20180615_WaterExtent_Assam_Sylhet0000000000-0000025088",
    "ST20180615_WaterExtent_Assam_Sylhet0000012544-0000000000",
    "ST20180615_WaterExtent_Assam_Sylhet0000012544-0000012544",
    "ST20180615_WaterExtent_Assam_Sylhet0000012544-0000025088",
    "ST_20170302_WaterExtent_OmusatiAndOshana0000000000-0000000000",
    "ST_20170302_WaterExtent_OmusatiAndOshana0000000000-0000012544",
    "ST20180210_WaterExtent_Cochabamba",
    "RS20180209_WaterExtent_Trinidad",
    "FL_20151015_SOM_20151116_Flood_Radarsat2",
    "KS5_20180729_WaterExtent_SanamxayDistrict",
    "KS5_20180730_WaterExtent_SamakkhixayDistrict",
    "LS20161216_Water_Extent_LakeChad0000025088-0000000000",
    "LS8_20160808_WaterExtent_WetSoil_Kassala0000012544-0000000000",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000025088-0000037632",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000025088-0000025088",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000025088-0000012544",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000012544-0000025088",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000012544-0000037632",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000012544-0000012544",
    "RS20180509_WaterExtent_Bulobarde_Saacow0000025088-0000000000",
    "RS2_20170324_WaterExtent_Piura",
    "RS2_20160812_WaterExtent_WetSoil_Sennar0000000000-0000012544",
    "RS2_20160812_WaterExtent_WetSoil_Sennar0000000000-0000000000",
    "RS2_20170217_WaterExtent_SaveRiver0000012544-0000000000",
    "RS2_20170217_WaterExtent_SaveRiver0000000000-0000012544",
    "RS2_20170217_WaterExtent_SaveRiver0000012544-0000012544",
    "ST1_20161024_WaterExtent_WetSoil_HaTinh_NorthernQuangBinh0000000000-0000012544",
    "ST1_20161024_WaterExtent_WetSoil_HaTinh_NorthernQuangBinh0000000000-0000000000",
    "ST1_20161024_WaterExtent_WetSoil_HaTinh_NorthernQuangBinh0000000000-0000012544",
    "ST1_20161024_WaterExtent_WetSoil_HaTinh_NorthernQuangBinh0000012544-0000000000",
    "ST1_20161024_WaterExtent_WetSoil_HaTinh_NorthernQuangBinh0000012544-0000012544",
    "ST1_20161012_WaterExtent_WetSoil_NorthernQuangBinh_HaTinh0000000000-0000000000",
    "RS2_20170912_WaterExtent_TurksAndCaicos",
    "RS2_20171005_WaterExtent_Bongor0000000000-0000012544",
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000000000-0000000000",
    "ST1_20161213_WaterExtent_Binhdinh_Phuyen0000000000-0000000000",
    "ST1_20161206_WaterExtent_ThuaThienHue0000012544-0000012544",
    "ST1_20161201_WaterExtent_Binhdinh_Phuyen_QuangNam0000012544-0000012544",
    "ST1_20161107_WaterExtent_BinhDinh_Lake_Phuyen_QuangNam0000000000-0000000000",
    "ST_20180508_WaterExtent_CumulativeKenya0000075264-0000025088",
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000012544-0000012544",
    "ST1_20161206_WaterExtent_ThuaThienHue0000000000-0000000000",
    "ST1_20161201_WaterExtent_Binhdinh_Phuyen_QuangNam0000037632-0000012544",
    "ST1_20161201_WaterExtent_Binhdinh_Phuyen_QuangNam0000037632-0000000000",
    "ST20180501_WaterExtent_MiddleJuba0000012544-0000012544", # almost
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000012544-0000000000"
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000012544-0000012544",
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000025088-0000000000",
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000025088-0000012544",
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000037632-0000012544",
    "ST1_20161107_WaterExtent_BinhDinh_Lake_Phuyen_QuangNam0000000000-0000012544",
    "ST1_20161107_WaterExtent_BinhDinh_Lake_Phuyen_QuangNam0000012544-0000012544",
    "ST1_20161201_WaterExtent_Binhdinh_Phuyen_QuangNam0000000000-0000000000",
    "ST1_20161201_WaterExtent_Binhdinh_Phuyen_QuangNam0000000000-0000012544",
    "ST_20160823_Water_Extent_LouangPrabang0000000000-0000000000",
    "ST_20160823_Water_Extent_LouangPrabang0000012544-0000000000",
    "ST_20160823_Water_Extent_LouangPrabang0000012544-0000012544",
    "ST_20170302_WaterExtent_OmusatiAndOshana0000012544-0000000000",
    "ST_20170730_WaterExtent_Champasack_Province0000000000-0000000000",
    "ST_20170812_WaterExtent_CentralBangladesh0000000000-0000012544",
    "ST1_20161206_WaterExtent_ThuaThienHue0000000000-0000012544",
    "ST20170104_WaterExtent_Narathiwat0000000000-0000012544",
    "ST20170104_WaterExtent_Narathiwat0000012544-0000012544",
    "ST1_20180928_WaterExtent_Bayelsa0000000000-0000000000",
    "ST1_20181003_WaterExtent_Bayelsa0000000000-0000000000",
    "ST1_20180928_WaterExtent_Bayelsa0000000000-0000012544",
    "ST20160630_WaterExtent_Dhaka_Rajshahi0000000000-0000000000",
    "ST20160630_WaterExtent_Dhaka_Rajshahi0000000000-0000012544",
    "ST20160630_WaterExtent_Dhaka_Rajshahi0000012544-0000000000",
    "ST20160630_WaterExtent_Dhaka_Rajshahi0000012544-0000012544",
    "ST20160714_WaterExtent_Rakhine0000000000-0000000000",
    "ST20160714_WaterExtent_Rakhine0000012544-0000000000",
    "ST20160714_WaterExtent_Rakhine0000012544-0000012544",
    "ST20160714_WaterExtent_Rakhine0000025088-0000000000",
    "ST1_20170105_WaterExtent_Wet_Soil_PungweRiver0000000000-0000000000", # almost
    "ST1_20170105_WaterExtent_Wet_Soil_PungweRiver0000000000-0000012544",
    "ST1_20170105_WaterExtent_Wet_Soil_PungweRiver0000012544-0000012544",
    "ST1_20170105_WaterExtent_Wet_Soil_PungweRiver0000012544-0000000000", # almost
    "ST1_20170129_WaterExtent_Wet_Soil_PungweRiver0000000000-0000000000", # almost
    "ST1_20170129_WaterExtent_Wet_Soil_PungweRiver0000000000-0000012544",
    "ST1_20170129_WaterExtent_Wet_Soil_PungweRiver0000012544-0000000000",
    "ST1_20170129_WaterExtent_Wet_Soil_PungweRiver0000012544-0000012544",
    "ST1_20180916_WaterExtent_Bayelsa0000000000-0000000000",
    "ST1_20180916_WaterExtent_Bayelsa0000000000-0000012544",
    "ST1_20180921_WaterExtent_Bayelsa0000000000-0000012544",
    "ST1_20180922_WaterExtent_Bayelsa0000000000-0000012544",
    "ST2_20181002_WaterExtent_Taipa",
    "ST20160724_WaterExtent_Dhaka_Rajshahi0000000000-0000000000",
    "ST_20170326_WaterExtent_OmusatiAndOshana0000012544-0000025088",
    "ST_20170302_WaterExtent_OmusatiAndOshana0000012544-0000012544",
    "ST_20170530_WaterExtent_Chittagong0000000000-0000000000",
    "TSX_20170310_WaterExtent_Maroantsetra0000000000-0000012544", #almost
    "TSX_20170310_WaterExtent_Maroantsetra0000012544-0000012544",
    "ST_20170607_WaterExtent_MaguindanaAndCotabatoProvinces0000000000-0000000000",
    "ST_20170607_WaterExtent_MaguindanaAndCotabatoProvinces0000000000-0000012544",
    "ST_20170607_WaterExtent_MaguindanaAndCotabatoProvinces0000012544-0000000000",
    "ST_20170607_WaterExtent_MaguindanaAndCotabatoProvinces0000012544-0000012544",
    "ST_20170822_WaterExtent_CentralSouthernBangladesh0000012544-0000000000",
    "ST_20171009_WaterExtent_Sayaxche0000012544-0000012544",
    "ST_20180508_WaterExtent_CumulativeKenya0000025088-0000037632",
    "ST_20180508_WaterExtent_CumulativeKenya0000037632-0000037632",
    "ST_20180508_WaterExtent_CumulativeKenya0000050176-0000012544",
    "ST_20180508_WaterExtent_CumulativeKenya0000050176-0000037632",
    "ST_20180508_WaterExtent_CumulativeKenya0000062720-0000000000",
    "ST_20180508_WaterExtent_CumulativeKenya0000062720-0000012544",
    "ST_20180508_WaterExtent_CumulativeKenya0000062720-0000025088",
    "ST_20180508_WaterExtent_CumulativeKenya0000062720-0000037632",
    "ST_20180508_WaterExtent_CumulativeKenya0000075264-0000000000",
    "ST_20180508_WaterExtent_CumulativeKenya0000075264-0000012544",
    "ST_20180508_WaterExtent_CumulativeKenya0000075264-0000037632",
]

FILENAMES_ERRORS_V1 = [
    "09262016_Cedar_River_at_Vinton_IA0000000000-0000000000",
    "09262016_Cedar_River_at_Vinton_IA0000000000-0000012544",
    "09262016_Cedar_River_at_Vinton_IA0000012544-0000000000",
    "09262016_Cedar_River_at_Vinton_IA0000012544-0000012544",
    "09262016_Iowa_River_at_Belle_Plaine_IA0000000000-0000000000",
    "09262016_Iowa_River_at_Belle_Plaine_IA0000000000-0000012544",
    "09262016_Mississippi_River_at_Clinton_IA0000000000-0000000000",
    "09262016_Wapsipinicon_River_at_Independence_IA0000000000-0000000000",
    "09262016_Wapsipinicon_River_at_Independence_IA0000000000-0000012544",
    "10132016_Ashley_River_near_North_Charleston_SC",
    "10132016_Cooper_River_at_Monks_Corner_SC",
    "10132016_Ashley_River_near_North_Charleston_SC",
    "EMSR258_06VLORE_DEL_v2_observed_event_a",
    "EMSR260_03SORBOLO_GRA_v3_observed_event_a",
    "EMSR260_04SANTILARIODENZA_GRA_v3_observed_event_a",
    "EMSR260_05MONTECCHIOEMILIA_GRA_v3_observed_event_a",
    "EMSR260_06SANPOLODENZA_DEL_MONIT01_v3_observed_event_a",
    "EMSR260_06SANPOLODENZA_DEL_v2_observed_event_a",
    "EMSR260_07CAMPOGALLIANO_DEL_v1_observed_event_a",
    "EMSR264_12FARAFANGANA_DEL_v2_observed_event_a",
    "EMSR264_14VANGAINDRANO_DEL_v2_observed_event_a",
    "EMSR264_18MIANDRIVAZODETAIL_DEL_v2_observed_event_a",
    "EMSR265_04ESBLY_DEL_MONIT06_v2_observed_event_a",
    "EMSR273_04GOMSIQE_DEL_MONIT04_v1_observed_event_a",
    "EMSR274_01AMBILOBE_DEL_v2_observed_event_a",
    "EMSR274_12ANTALAHA_DEL_v1_observed_event_a",
    "EMSR274_14MAHAJANGA_DEL_v1_observed_event_a",
    "EMSR277_02DIDIMOTICHO_DEL_MONIT01_v2_observed_event_a",
    "EMSR277_02DIDIMOTICHO_DEL_v2_observed_event_a",
    "EMSR277_03FERES_DEL_MONIT01_v1_observed_event_a",
    "EMSR277_03FERES_DEL_v1_observed_event_a",
    "EMSR311_03MYRTLEBEACH_DEL_MONIT01_v1_observed_event_a",
    "EMSR311_04WILMINGTON_DEL_MONIT01_v1_observed_event_a",
    "EMSR311_05JACKSONVILLE_DEL_MONIT03_v1_observed_event_a",
    "EMSR312_01LAOAG_GRA_v1_observed_event_a",
    "EMSR312_07VIGAN_DEL_MONIT01_v1_observed_event_a",
    "EMSR312_07VIGAN_DEL_v1_observed_event_a",
    "EMSR312_08CANDON_DEL_MONIT01_v1_observed_event_a",
    "EMSR312_08CANDON_DEL_v1_observed_event_a",
    "EMSR322_17MARIANNA_DEL_v1_observed_event_a",
    "EMSR329_05CASTIADAS_GRA_v3_observed_event_a",
    "EMSR333_12BAGHERIA_DEL_v1_observed_event_a",
    "EMSR339_05BEIRASE_DEL_v1_observed_event_a",
    "EMSR346_03BEIRA_DEL_v1_observed_event_a",
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000012544-0000000000"
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000012544-0000012544",
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000025088-0000000000",
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000025088-0000012544",
    "RS_20170205_WaterExtent_Chimuara_SaveRiver0000037632-0000012544",
    "ST1_20170105_WaterExtent_Wet_Soil_PungweRiver0000000000-0000000000",
    "ST1_20170105_WaterExtent_Wet_Soil_PungweRiver0000000000-0000012544",
    "ST1_20170105_WaterExtent_Wet_Soil_PungweRiver0000012544-0000000000",
    "ST1_20170129_WaterExtent_Wet_Soil_PungweRiver0000000000-0000000000",
    "ST1_20170129_WaterExtent_Wet_Soil_PungweRiver0000000000-0000012544",
    "ST1_20170129_WaterExtent_Wet_Soil_PungweRiver0000012544-0000000000",
    "ST1_20170129_WaterExtent_Wet_Soil_PungweRiver0000012544-0000012544",
    "ST1_20180916_WaterExtent_Bayelsa0000000000-0000000000",
    "ST1_20180916_WaterExtent_Bayelsa0000000000-0000012544",
    "ST20170104_WaterExtent_Narathiwat0000000000-0000012544",
    "ST20170104_WaterExtent_Narathiwat0000012544-0000012544",
    "ST20180501_WaterExtent_MiddleJuba0000012544-0000012544",
    "ST2_20181002_WaterExtent_Taipa",
    "ST_20170302_WaterExtent_OmusatiAndOshana0000012544-0000012544",
    "ST_20170426_WaterExtent_LesCayes_Cavaillon",
    "TSX_20170308_WaterExtent_Maroantsetra",
    "TSX_20170310_WaterExtent_Maroantsetra0000000000-0000012544",
    "TSX_20170310_WaterExtent_Maroantsetra0000012544-0000012544"
]


VAL_FILENAMES = ["RS2_20161008_Water_Extent_Corail_Pestel",
                 "ST1_20161014_WaterExtent_BinhDinh_Lake",
                 "EMSR271_02FARKADONA_DEL_v1_observed_event_a",
                 "EMSR287_05MAGWITZ_DEL_v1_observed_event_a",
                 "EMSR280_03FALUN_DEL_MONIT06_v2_observed_event_a",
                 "EMSR279_05ZARAGOZA_DEL_MONIT02_v1_observed_event_a"]