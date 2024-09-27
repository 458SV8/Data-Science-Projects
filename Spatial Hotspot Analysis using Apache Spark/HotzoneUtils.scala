package cse512

object HotzoneUtils {
  def ST_Contains(queryRectangle: String, pointString: String): Boolean = {
    val Array(x1, y1, x2, y2) = queryRectangle.split(",").map(x => x.toDouble)
    val xmin = math.min(x1, x2)
    val xmax = math.max(x1, x2)
    val ymin = math.min(y1, y2)
    val ymax = math.max(y1, y2)
    val Array(xp, yp) = pointString.split(',').map(x => x.toDouble)
    (xp >= xmin) && (xp <= xmax) && (yp >= ymin) && (yp <= ymax)
  }
}
