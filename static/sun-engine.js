/* Pure sunrise/sunset (NOAA "Almanac for Computers"). No DOM, no network.
   Browser: window.RMSun. Node: module.exports. */
(function (root, factory) {
  var api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') window.RMSun = api;
})(this, function () {
  var RAD = Math.PI / 180;

  // Returns local clock hours (0..24) for the event, or null if the sun does
  // not cross the horizon that day at that latitude.
  function sunEvent(date, lat, lng, isSunrise) {
    var start = new Date(date.getFullYear(), 0, 0);
    var dayOfYear = Math.floor((date - start) / 86400000);
    var lngHour = lng / 15;
    var t = dayOfYear + ((isSunrise ? 6 : 18) - lngHour) / 24;
    var M = 0.9856 * t - 3.289;
    var L = M + 1.916 * Math.sin(M * RAD) + 0.020 * Math.sin(2 * M * RAD) + 282.634;
    L = (L % 360 + 360) % 360;
    var RA = Math.atan(0.91764 * Math.tan(L * RAD)) / RAD;
    RA = (RA % 360 + 360) % 360;
    RA += (Math.floor(L / 90) * 90) - (Math.floor(RA / 90) * 90);
    RA /= 15;
    var sinDec = 0.39782 * Math.sin(L * RAD);
    var cosDec = Math.cos(Math.asin(sinDec));
    var cosH = (Math.cos(90.833 * RAD) - sinDec * Math.sin(lat * RAD)) / (cosDec * Math.cos(lat * RAD));
    if (cosH > 1 || cosH < -1) return null;
    var H = isSunrise ? 360 - Math.acos(cosH) / RAD : Math.acos(cosH) / RAD;
    H /= 15;
    var T = H + RA - 0.06571 * t - 6.622;
    var UT = ((T - lngHour) % 24 + 24) % 24;
    var offsetHours = -date.getTimezoneOffset() / 60; // local offset of the host
    return ((UT + offsetHours) % 24 + 24) % 24;
  }

  function sunTimes(date, lat, lng) {
    return { sunrise: sunEvent(date, lat, lng, true), sunset: sunEvent(date, lat, lng, false) };
  }

  return { sunEvent: sunEvent, sunTimes: sunTimes };
});
