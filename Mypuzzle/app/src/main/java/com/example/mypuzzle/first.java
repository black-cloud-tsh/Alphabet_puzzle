package com.example.mypuzzle;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.content.Intent;

public class first extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_first);
    }

    public void back(View view) {
        Intent intent = new Intent();
        intent.setClass(first.this,MainActivity.class);
        startActivity(intent);
    }
}