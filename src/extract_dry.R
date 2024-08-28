library(rvest)
library(raster)

url <- "https://edcftp.cr.usgs.gov/project/vegdyn/vegdri/Geotif/"
doc <- read_html(url)
links <- html_attr(html_nodes(doc, "a"), "href")
links <- links[grepl("22.tif", links)]
free(doc)

weeks_2022 <- c(
  31,  # Scottsbluff (TP3)
  31,  # Lincoln (TP2)
  29,  # MOValley (TP2)
  31,  # Ames (TP3). Obs: no week32 so let's use 33
  31   # Crawfordsville (TP3)
)
df_weeks_2022 <- data.frame(location = c("Scottsbluff", "Lincoln", "MOValley", "Ames", "Crawfordsville"))
df_weeks_2022$week <- paste0("week", weeks_2022)
for (i in 1:nrow(df_weeks_2022)) {
  week <- df_weeks_2022[i, "week"]
  location <- df_weeks_2022[i, "location"]
  link <- links[grep(week, links)]
  download.file(link, paste0("output/", "vegdry_", location, "_2022"))
}
