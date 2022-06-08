using UnityEngine;
using System;
using System.Text;
using UnityEngine.UI;


public class Client : MonoBehaviour
{
    string recordingsString;
    Requester requester;
    public GameObject dataRecorder;

    private void Start()
    {
        requester = new Requester();
        requester.recordingsString = null;
        requester.Start();
        recordingsString = null;

    }


    // Update is called once per frame
    void Update()
    {
        recordingsString = dataRecorder.GetComponent<DataRecorder>().recordingsString;
        if (recordingsString != null)
            requester.recordingsString = recordingsString;        
    }


    private void OnDestroy()
    {
        requester.Stop();
    }
}