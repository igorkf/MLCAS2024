import os

import requests
from bs4 import BeautifulSoup
import rasterio
from rasterio.warp import transform_bounds


def getval(img, lon, lat):
    idx = img.index(lon, lat, precision=1E-6)    
    return img[idx]


if __name__ == "__main__":
    site = "https://edcftp.cr.usgs.gov/project/vegdyn/vegdri/Geotif"
    response = requests.get(site)
    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all("a")
    urls = [x.text for x in img_tags if ".tif" in x.text]

    # 2022
    w2022 = [
        (31, "Scottsbluff"), #41.85, 103.70),  # TP3
        (31, "Lincoln"),  # TP2
        (29, "MOValley"),  # TP2
        (31, "Ames"),  # TP3. Obs: no week32 so let's use 31
        (31, "Crawfordsville"),  # TP3
    ]
    for week, location in w2022:
        match = [x for x in urls if f"week{week}" in x][0]  # overkill
        filename = f"output/vegdry_{location}.tif"
        with open(filename, "wb") as f:
            response = requests.get(os.path.join(site, match))
            f.write(response.content)
        with rasterio.open(filename, "r+") as src:
            print("")
            bounds = (-90, -180, 90, 180)
            xmin, ymin, xmax, ymax = transform_bounds(src.crs, 4326, *bounds)
            idx = src.index(41.85, 103.7)
            img = src.read()[0]
            val = getval()

# doc <- read_html(url)
# links <- html_attr(html_nodes(doc, "a"), "href")
# links <- links[grepl("22.tif", links)]
# free(doc)

# weeks_2022 <- c(
#   31,  # Scottsbluff (TP3)
#   31,  # Lincoln (TP2)
#   29,  # MOValley (TP2)
#   31,  # Ames (TP3). Obs: no week32 so let's use 33
#   31   # Crawfordsville (TP3)
# )
# df_weeks_2022 <- data.frame(location = c("Scottsbluff", "Lincoln", "MOValley", "Ames", "Crawfordsville"))
# df_weeks_2022$week <- paste0("week", weeks_2022)
# for (i in 1:nrow(df_weeks_2022)) {
#   week <- df_weeks_2022[i, "week"]
#   location <- df_weeks_2022[i, "location"]
#   link <- links[grep(week, links)]
#   download.file(link, paste0("output/", "vegdry_", location, "_2022"))
# }
