from tile2net import Raster, Grid, PedNet
import pprint
import geopandas as gpd

polys = gpd.read_file(r'C:\Users\MC\Downloads\Da\Project_NYC\polygons\Polygons-WSQ-08-03-2023_14-51')
# directory to save the project
output_dir = r'C:\Users\MC\Downloads\Da'
# region of interest to get its bounding box
location_name = 'Washington Square Park, NYC, NY'
raster = Raster(
    name='Project_NYC',  # project name
    zoom=19,
    base_tilesize=256,
    location=location_name,
    source='ny',  # download sources - check current sources at ....
    output_dir=output_dir,
    # stitch_step=1
)
g_made2 = Grid(name='Project_NYC',
              location=location_name, project= raster.project, stitch_step=4)
raster.generate(4)  # stitch every 16 tiles, (4 columns x 4 rows)
# print(raster.project)
net = PedNet(polys, g_made2.project)
net.convert_whole_poly2line()
print('\n\n *************** json Info ***************\n')
pprint.pprint(raster.save_info_json(return_dict=True))
raster.project.structure.dump()