package com.example.mypuzzle;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.content.Intent;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    public void begins(View view) {
            Intent intent = new Intent();
            intent.setClass(MainActivity.this,Game.class);
            startActivity(intent);
    }

    public void history(View view) {
        Intent intent = new Intent();
        intent.setClass(MainActivity.this,History.class);
        startActivity(intent);
    }
}