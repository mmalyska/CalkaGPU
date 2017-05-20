using Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;

namespace CalkaRestApi.Controllers
{
    public class CalkaController : ApiController
    {
        [HttpGet]
        [Route("api/calka/wynik/cpu")]
        public IHttpActionResult WynikCPU(double lb, double ub, int n)
        {
            return Ok((new CalkaHelper()).CalkaCPU(lb, ub, n));
        }

       
    }
}
