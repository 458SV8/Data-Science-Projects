package cse512

import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object Entrance extends App {
  override def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .appName("CSE512-HotspotAnalysis-NUTELLA")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()
    paramsParser(spark, args)
  }
  private def paramsParser(spark: SparkSession, args: Array[String]): Unit = {
    var paramOffset = 1
    var currentQueryParams = ""
    var currentQueryName = ""
    var currentQueryIdx = -1
    while (paramOffset <= args.length) {
      if (paramOffset == args.length || args(paramOffset).toLowerCase.contains("analysis")) {
        if (currentQueryIdx != -1) queryLoader(spark, currentQueryName, currentQueryParams, args(0) + currentQueryIdx)
        if (paramOffset == args.length) return
        currentQueryName = args(paramOffset)
        currentQueryParams = ""
        currentQueryIdx = currentQueryIdx + 1
      }
      else {
        currentQueryParams = currentQueryParams + args(paramOffset) + " "
      }
      paramOffset = paramOffset + 1
    }
  }
  private def queryLoader(spark: SparkSession, queryName: String, queryParams: String, outputPath: String) {
    val queryParam = queryParams.split(" ")
    if (queryName.equalsIgnoreCase("hotcellanalysis")) {
      HotcellAnalysis.runHotcellAnalysis(spark, queryParam(0)).limit(50).write.mode(SaveMode.Overwrite).csv(outputPath)
    }
    else if (queryName.equalsIgnoreCase("hotzoneanalysis")) {
      HotzoneAnalysis.runHotZoneAnalysis(spark, queryParam(0), queryParam(1)).write.mode(SaveMode.Overwrite).csv(outputPath)
    }
    else {
      throw new NoSuchElementException("[CSE512] The given query name " + queryName + " is wrong.")
    }
  }
}
