using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class CalculateHeadVelocity : MonoBehaviour
{
    Camera headCamera;
    // head rotation velocity in the Longitude direction.
    public float headVelX;
    // head rotation velocity in the Latitude direction.
    public float headVelY;
    bool firstFrame = true;
    long time_last;
    float headLongitude_last;
    float headLatitude_last;


    void Start()
    {
        headCamera = this.GetComponent<Camera>();
        headVelX = 0;
        headVelY = 0;
    }

    // Update is called once per frame
    void Update()
    {
        // Timestamp
        System.TimeSpan timeSpan = System.DateTime.Now - new System.DateTime(1970, 1, 1, 0, 0, 0);
        long time_current = (long)timeSpan.TotalMilliseconds - 8 * 60 * 60 * 1000;

        // Head Orientation
        float x = headCamera.transform.forward.x;
        float y = headCamera.transform.forward.y;
        float z = headCamera.transform.forward.z;
        float xz = Mathf.Sqrt(Mathf.Pow(x, 2) + Mathf.Pow(z, 2));
        float headLongitude_current = Mathf.Atan2(x, z);
        float headLatitude_current = Mathf.Atan2(y, xz);
       

        if (firstFrame)
        {
            time_last = time_current;
            headLongitude_last = headLongitude_current;
            headLatitude_last = headLatitude_current;
            firstFrame = false;
        }
        else
        {
            headVelX = CalculateAngularVelocity(headLongitude_current, headLongitude_last, time_current, time_last);
            headVelY = CalculateAngularVelocity(headLatitude_current, headLatitude_last, time_current, time_last);
            time_last = time_current;
            headLongitude_last = headLongitude_current;
            headLatitude_last = headLatitude_current;

            //Debug.Log("Head Velocity: " + headVelX + "," + headVelY);       
        }
    }

    // Calculate angular velocity (deg/s) using small-angle approximation
    float CalculateAngularVelocity(float ang1, float ang2, long time1, long time2)
    {
        //the range of the velocity.
        float velRange = 700f;
        float vel = Mathf.Sin(ang1 - ang2) / Mathf.PI * 180/(time1-time2)*1000;

        if (vel > velRange)
            return velRange;
        if (vel < -velRange)
            return -velRange;
        return vel;
    }
}
