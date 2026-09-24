package com.example.ewastetester

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : Activity() {

    private lateinit var interpreter: Interpreter
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView

    private val classNames = arrayOf(
        "Battery",
        "Cable",
        "Keyboard",
        "Microwave",
        "Mobile",
        "Mouse",
        "PCB",
        "Player",
        "Printer",
        "Television",
        "Washing Machine"
    )

    private val inputSize = 224
    private val cameraRequest = 1001

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.resultText)

        interpreter = Interpreter(loadModel())

        findViewById<Button>(R.id.cameraButton).setOnClickListener {
            openCamera()
        }
    }

    private fun loadModel(): ByteBuffer {

        val fileDescriptor =
            assets.openFd("ewaste_mobilenetv2_fp16.tflite")

        val inputStream =
            fileDescriptor.createInputStream()

        val bytes = inputStream.readBytes()

        val buffer =
            ByteBuffer.allocateDirect(bytes.size)

        buffer.order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        buffer.rewind()

        inputStream.close()

        return buffer
    }

    private fun openCamera() {

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {

            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                cameraRequest
            )

            return
        }

        val intent =
            Intent(MediaStore.ACTION_IMAGE_CAPTURE)

        startActivityForResult(
            intent,
            cameraRequest
        )
    }

    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?
    ) {

        super.onActivityResult(
            requestCode,
            resultCode,
            data
        )

        if (
            requestCode == cameraRequest &&
            resultCode == RESULT_OK
        ) {

            val bitmap =
                data?.extras?.get("data") as? Bitmap

            if (bitmap != null) {

                imageView.setImageBitmap(bitmap)

                classify(bitmap)
            }
        }
    }

    private fun classify(bitmap: Bitmap) {

        val resized =
            Bitmap.createScaledBitmap(
                bitmap,
                inputSize,
                inputSize,
                true
            )

        val inputBuffer =
            ByteBuffer.allocateDirect(
                1 *
                inputSize *
                inputSize *
                3 *
                4
            )

        inputBuffer.order(
            ByteOrder.nativeOrder()
        )

        val pixels =
            IntArray(
                inputSize *
                inputSize
            )

        resized.getPixels(
            pixels,
            0,
            inputSize,
            0,
            0,
            inputSize,
            inputSize
        )

        for (pixel in pixels) {

            val r =
                (pixel shr 16) and 0xFF

            val g =
                (pixel shr 8) and 0xFF

            val b =
                pixel and 0xFF

            /*
             * IMPORTANT:
             * The trained model expects RAW 0-255
             * float32 input.
             *
             * DO NOT normalize to [-1, 1].
             */

            inputBuffer.putFloat(r.toFloat())
            inputBuffer.putFloat(g.toFloat())
            inputBuffer.putFloat(b.toFloat())
        }

        inputBuffer.rewind()

        val output =
            Array(1) {
                FloatArray(classNames.size)
            }

        interpreter.run(
            inputBuffer,
            output
        )

        var bestIndex = 0
        var bestConfidence = output[0][0]

        for (i in 1 until classNames.size) {

            if (output[0][i] > bestConfidence) {

                bestConfidence =
                    output[0][i]

                bestIndex = i
            }
        }

        val percentage =
            bestConfidence * 100f

        resultText.text =
            "${classNames[bestIndex]}\n\n" +
            "Confidence: %.2f%%"
                .format(percentage)
    }

    override fun onDestroy() {

        interpreter.close()

        super.onDestroy()
    }
}
