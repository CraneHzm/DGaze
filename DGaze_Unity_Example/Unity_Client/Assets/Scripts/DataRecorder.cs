using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class DataRecorder : MonoBehaviour
{
    public GameObject DynamicObjects;
    public Camera headCamera;
    // the number of data item in one recording
    public int dataNumber = 50;
    // time rate to sample the data (hz)
    float sampleRate = 100;
    // time offset (ms) used to correct the sampling time, it depends on how fast your machine runs
    float timeOffset = 0;
    Queue<string> headObjectData;
    public string recordingsString;
    public bool Running;

    // Start is called before the first frame update
    void Start()
    {
        headObjectData = new Queue<string>();
        recordingsString = null;
        Running = true;
        StartCoroutine(RecordData());
    }


    IEnumerator RecordData()
    {
        WaitForSecondsRealtime waitTime = new WaitForSecondsRealtime(1f/sampleRate - timeOffset/1000);

        while (Running)
        {
            SampleData();
            yield return waitTime;
        }
    }

    void SampleData()
    {
        // Timestamp
        System.TimeSpan timeSpan = System.DateTime.Now - new System.DateTime(1970, 1, 1, 0, 0, 0);
        long time = (long)timeSpan.TotalMilliseconds - 8 * 60 * 60 * 1000;
        string timeStamp = time.ToString();


        // Head Rotation Velocity
        float headVelX = headCamera.GetComponent<CalculateHeadVelocity>().headVelX;
        float headVelY = headCamera.GetComponent<CalculateHeadVelocity>().headVelY;
        string trackedObjectsString = DynamicObjects.GetComponent<TrackObjects>().trackedObjectsString;
        string info = headVelX.ToString("f2") + "," + headVelY.ToString("f2") + "," + trackedObjectsString;

        if (headObjectData.Count < dataNumber)
        {            
            headObjectData.Enqueue(info);
            if (headObjectData.Count == dataNumber)
            {
                // Collect data in time ascending order (t- Delta t, ..., t-10, t).
                recordingsString = timeStamp;
                foreach (string data in headObjectData)
                    recordingsString += "," + data;                                     
                //Debug.Log(recordingsString);
            }

        }
        else if (headObjectData.Count == dataNumber)
        {                                      
            // renew the info
            headObjectData.Dequeue();
            headObjectData.Enqueue(info);
            // Collect data in time ascending order (t- Delta t, ..., t-10, t).
            recordingsString = timeStamp;
            foreach (string data in headObjectData)
                recordingsString += "," + data;
            //Debug.Log(recordingsString);
        }
    }


}
