using System;
using System.Collections.Generic;
using System.Linq;

namespace Emeocv_Sharp
{
    public class Plausi
    {
        Dictionary<DateTime, double> _queue;
        DateTime _time;
        double _value;
        double _maxPower;
        int _window;

        public Plausi(double maxPower = 50 /*kW*/, int window = 13)
        {
            _maxPower = maxPower;
            _window = window;
            _value = -1;
            _time = DateTime.Now;
        }

        public double CheckedValue
        {
            get { return _value; }
        }

        DateTime CheckedTime
        {
            get { return _time; }
            
        }

        string QueueAsString()
        {
            string str = string.Empty;
            str += "[";

            foreach (var item in _queue)
            {
                str += string.Format("%.1f", item.Value);
                str += ", ";
            }

            str += "]";
            return str;
        }

        public bool Check(string value, DateTime time)
        {
            if (value.Length != 7)
            {
                // exactly 7 digits
                Console.WriteLine("Plausi rejected: exactly 7 digits");
                return false;
            }
            if (value.IndexOf('?') == -1)
            {
                // no '?' char
                Console.WriteLine("Plausi rejected: no '?' char");
                return false;
            }

            double dval = double.Parse(value) / 10;
            _queue.Add(time, dval);

            if (_queue.Count < _window)
            {
                Console.WriteLine("Plausi rejected: not enough values: %d", _queue.Count());
                return false;
            }
            if (_queue.Count() > _window)
            {
                _queue.Remove(_queue.FirstOrDefault().Key);
            }

            // iterate through queue and check that all values are ascending
            // and consumption of energy is less than limit
            for (int i = 0; i < _queue.Count; i++)
            {
                var dval_ = i > 0 ? _queue.ElementAt(i - 1).Value : i;

                if (_queue.ElementAt(i).Value < dval)
                {
                    // value must be >= previous value
                    Console.WriteLine("Plausi rejected: value must be >= previous value");
                    return false;
                }

                var power_ = (_queue.ElementAt(i).Value - dval) / (_queue.ElementAt(i).Key - time).TotalSeconds * 3600;
                if (power_ > _maxPower)
                {
                    // consumption of energy must not be greater than limit
                    Console.WriteLine("Plausi rejected: consumption of energy %.3f must not be greater than limit %.3f", power_, _maxPower);
                    return false;
                }
                time = _queue.ElementAt(i).Key;
                dval = _queue.ElementAt(i).Value;
            }

            var candTime = _queue.FirstOrDefault(i => i.Value == _window / 2).Key;
            double candValue = _queue.Where(i => i.Value == _window / 2).ElementAt(1).Value;
            if (candValue < _value)
            {
                Console.WriteLine("Plausi rejected: value must be >= previous checked value");
                return false;
            }
            double power = (candValue - _value) / (candTime - _time).TotalSeconds * 3600;
            if (power > _maxPower)
            {
                Console.WriteLine("Plausi rejected: consumption of energy (checked value) %.3f must not be greater than limit %.3f", power, _maxPower);
                return false;
            }

            // everything is OK -> use the candidate value
            _time = candTime;
            _value = candValue;
            return true;
        }
    }
}
